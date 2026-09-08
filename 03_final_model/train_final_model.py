#!/usr/bin/env python3
"""Train the final pointwise TCN using the best completed W&B sweep run.

The selected hyperparameters are fetched from W&B. All four airports are split
independently and stratified into training/validation flights, exactly as in the
hyperparameter-optimization pipeline. The scaler and class weights are fitted
only on training flights. One model is trained on the pooled training flights,
early stopped on the pooled validation flights, and the current core-only
postprocessing grid is optimized globally on validation predictions.

This directory is self-contained: it does not import code from ``HP Optimization``.
"""

from __future__ import annotations

# These variables must be set before torch/CUDA are imported.
import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from final_model_pipeline.config import (
    AIRPORT_FILES,
    AIRPORTS,
    BATCH_SIZE,
    CORE_MAX_INTERRUPTION_SECONDS,
    CORE_PROBABILITY_THRESHOLDS,
    DENSE_UNITS,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IOU_THRESHOLDS,
    LABEL_COL,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    MIN_CORE_SEGMENT_DURATION_SECONDS,
    MIN_LR,
    PROJECT_ROOT,
    RANDOM_STATE,
    TCN_DILATIONS,
    TCN_FILTERS,
    TCN_KERNEL_SIZE,
    VAL_SIZE,
)
from final_model_pipeline.data import (
    SequenceRecord,
    build_sequences_for_airport,
    create_airport_splits,
    fit_scaler,
    make_loader,
    transform_sequences,
)
from final_model_pipeline.evaluation import evaluate_best_postprocessing_for_airport
from final_model_pipeline.features import resolve_feature_configuration
from final_model_pipeline.model import PointwiseTCN, count_trainable_parameters
from final_model_pipeline.postprocessing import (
    build_postprocessing_cache,
    optimize_postprocessing,
)
from final_model_pipeline.runtime import (
    DEVICE,
    POSTPROCESSING_WORKERS,
    USE_AMP,
    reproducibility_metadata,
    seed_everything,
)
from final_model_pipeline.training import (
    FoldTrainingResult,
    calculate_class_weights,
    evaluate_pointwise,
    predict_probabilities,
    weighted_masked_loss,
)
from final_model_pipeline.utils import json_compatible, nanmean


DEFAULT_SOURCE_RUN = "hanv-zhaw/pushback-detection/c0typkx3"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_model"
)

REQUIRED_MODEL_CONFIG_KEYS = (
    "feature_variant",
    "track_window_preset",
    "learning_rate",
    "dropout",
    "positive_weight_scale",
)


def normalize_wandb_run_path(value: str) -> str:
    """Return ``entity/project/run_id`` from a W&B URL or direct run path."""

    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("The W&B source run may not be empty")

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if "runs" not in parts:
            raise ValueError("The W&B URL must contain '/runs/<run_id>'.")
        runs_index = parts.index("runs")
        if runs_index < 2 or runs_index + 1 >= len(parts):
            raise ValueError(f"Could not parse W&B run URL: {value}")
        entity = parts[0]
        project = parts[1]
        run_id = parts[runs_index + 1]
        return f"{entity}/{project}/{run_id}"

    parts = value.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Use a W&B URL or a path in the form entity/project/run_id."
        )
    return value


def fetch_source_run_config(
    source_run_path: str,
) -> tuple[Any, dict[str, Any]]:
    """Fetch the chosen sweep run and resolve its effective feature setup."""

    api = wandb.Api()
    source_run = api.run(source_run_path)
    source_config = dict(source_run.config)

    missing = [
        key for key in REQUIRED_MODEL_CONFIG_KEYS
        if key not in source_config
    ]
    if missing:
        raise KeyError(
            f"Source run {source_run_path} is missing config keys: {missing}"
        )

    feature_variant = str(source_config["feature_variant"])
    sampled_track_window_preset = str(source_config["track_window_preset"])
    (
        feature_setup,
        effective_track_window_preset,
        track_windows,
    ) = resolve_feature_configuration(
        feature_variant=feature_variant,
        sampled_track_window_preset=sampled_track_window_preset,
    )

    selected_config = {
        "feature_variant": feature_variant,
        "track_window_preset": sampled_track_window_preset,
        "sampled_track_window_preset": sampled_track_window_preset,
        "effective_track_window_preset": effective_track_window_preset,
        "track_windows": track_windows,
        "feature_setup": feature_setup,
        "learning_rate": float(source_config["learning_rate"]),
        "dropout": float(source_config["dropout"]),
        "positive_weight_scale": float(source_config["positive_weight_scale"]),
    }

    # Cross-check the effective values saved by the sweep when available.
    expected_setup = source_config.get("feature_setup")
    if expected_setup is not None and str(expected_setup) != feature_setup:
        raise RuntimeError(
            "The source run's saved feature_setup does not match the current "
            f"resolver: {expected_setup!r} != {feature_setup!r}"
        )

    expected_effective = source_config.get("effective_track_window_preset")
    if expected_effective != effective_track_window_preset:
        # Older W&B configs may omit this key entirely; only enforce if present.
        if "effective_track_window_preset" in source_config:
            raise RuntimeError(
                "The source run's effective track-window preset does not match "
                "the current resolver."
            )

    expected_windows = source_config.get("track_windows")
    if expected_windows is not None:
        if [int(v) for v in expected_windows] != [int(v) for v in track_windows]:
            raise RuntimeError(
                "The source run's effective track windows do not match the "
                "current resolver."
            )

    return source_run, selected_config


def build_final_splits(
    sequences_by_airport: dict[str, list[SequenceRecord]],
) -> tuple[
    dict[str, dict[str, list[int]]],
    dict[str, list[SequenceRecord]],
    dict[str, list[SequenceRecord]],
    list[SequenceRecord],
    list[SequenceRecord],
]:
    """Create deterministic per-airport splits and pooled train/validation."""

    split_indices = create_airport_splits(sequences_by_airport)
    train_by_airport: dict[str, list[SequenceRecord]] = {}
    validation_by_airport: dict[str, list[SequenceRecord]] = {}

    for airport in AIRPORTS:
        sequences = sequences_by_airport[airport]
        train_by_airport[airport] = [
            sequences[index]
            for index in split_indices[airport]["train"]
        ]
        validation_by_airport[airport] = [
            sequences[index]
            for index in split_indices[airport]["validation"]
        ]

    train_sequences = [
        sequence
        for airport in AIRPORTS
        for sequence in train_by_airport[airport]
    ]
    validation_sequences = [
        sequence
        for airport in AIRPORTS
        for sequence in validation_by_airport[airport]
    ]

    return (
        split_indices,
        train_by_airport,
        validation_by_airport,
        train_sequences,
        validation_sequences,
    )


def build_split_dataframe(
    train_by_airport: dict[str, list[SequenceRecord]],
    validation_by_airport: dict[str, list[SequenceRecord]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name, source in (
        ("train", train_by_airport),
        ("validation", validation_by_airport),
    ):
        for airport in AIRPORTS:
            for sequence in source[airport]:
                records.append(
                    {
                        "split": split_name,
                        "airport": airport,
                        "flight_id": sequence.flight_id,
                        "has_pushback": sequence.has_pushback,
                        "length": sequence.length,
                        "n_pushback_points": int(sequence.y.sum()),
                    }
                )
    return pd.DataFrame(records)


def train_final_model(
    train_sequences: list[SequenceRecord],
    validation_sequences: list[SequenceRecord],
    n_features: int,
    learning_rate: float,
    dropout: float,
    positive_weight_scale: float,
    run: Any,
) -> FoldTrainingResult:
    """Train one pooled all-airports model with the sweep's runtime logic."""

    train_loader = make_loader(
        train_sequences,
        shuffle=True,
        seed=RANDOM_STATE,
    )
    validation_loader = make_loader(
        validation_sequences,
        shuffle=False,
        seed=RANDOM_STATE + 1,
    )

    model = PointwiseTCN(
        n_features=n_features,
        filters=TCN_FILTERS,
        kernel_size=TCN_KERNEL_SIZE,
        dilations=TCN_DILATIONS,
        dense_units=DENSE_UNITS,
        dropout=dropout,
    ).to(DEVICE)

    weight_0, weight_1 = calculate_class_weights(
        train_sequences,
        positive_weight_scale,
    )
    class_weights = torch.tensor(
        [weight_0, weight_1],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_aucpr = -np.inf
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    early_stop_counter = 0
    lr_reduce_counter = 0
    history: list[dict[str, float]] = []

    run.define_metric("training/epoch")
    for metric_name in (
        "train_loss",
        "val_loss",
        "val_aucpr",
        "val_aucroc",
        "val_accuracy",
        "learning_rate",
    ):
        run.define_metric(
            f"training/{metric_name}",
            step_metric="training/epoch",
        )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=DEVICE)
        train_weight_sum = torch.zeros((), device=DEVICE)

        for X, y, mask, _, _ in train_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(X)
                loss = weighted_masked_loss(
                    criterion,
                    logits,
                    y,
                    mask,
                    class_weights,
                )

            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()

            batch_weight = mask.sum()
            train_loss_sum += loss.detach() * batch_weight
            train_weight_sum += batch_weight

        train_loss = float(
            (train_loss_sum / train_weight_sum.clamp_min(1.0)).item()
        )
        validation_metrics = evaluate_pointwise(
            model,
            validation_loader,
            criterion,
            class_weights,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])

        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_aucpr": validation_metrics["aucpr"],
            "val_aucroc": validation_metrics["aucroc"],
            "val_accuracy": validation_metrics["accuracy"],
        }
        history.append(record)
        run.log(
            {
                "training/epoch": epoch,
                "training/train_loss": train_loss,
                "training/val_loss": validation_metrics["loss"],
                "training/val_aucpr": validation_metrics["aucpr"],
                "training/val_aucroc": validation_metrics["aucroc"],
                "training/val_accuracy": validation_metrics["accuracy"],
                "training/learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={validation_metrics['loss']:.5f} | "
            f"val_aucpr={validation_metrics['aucpr']:.5f}",
            flush=True,
        )

        val_aucpr = validation_metrics["aucpr"]
        improved = not np.isnan(val_aucpr) and val_aucpr > best_val_aucpr

        if improved:
            best_val_aucpr = float(val_aucpr)
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
            lr_reduce_counter = 0
        else:
            early_stop_counter += 1
            lr_reduce_counter += 1

        if lr_reduce_counter >= LR_REDUCE_PATIENCE:
            old_lr = float(optimizer.param_groups[0]["lr"])
            new_lr = max(old_lr * LR_REDUCE_FACTOR, MIN_LR)
            if new_lr < old_lr:
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = new_lr
                print(
                    f"ReduceLROnPlateau: lr {old_lr:.2e} -> {new_lr:.2e}",
                    flush=True,
                )
            lr_reduce_counter = 0

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                f"best val_aucpr={best_val_aucpr:.5f}",
                flush=True,
            )
            break

    if best_state_dict is None:
        raise RuntimeError("No valid final-model checkpoint was produced")

    model.load_state_dict(best_state_dict)
    final_validation_metrics = evaluate_pointwise(
        model,
        validation_loader,
        criterion,
        class_weights,
    )

    return FoldTrainingResult(
        model=model,
        history=pd.DataFrame(history),
        best_epoch=best_epoch,
        best_val_aucpr=best_val_aucpr,
        final_val_metrics=final_validation_metrics,
        class_weight_0=weight_0,
        class_weight_1=weight_1,
    )


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def numeric_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "airport":
            continue
        if isinstance(value, (int, float, np.number)):
            output[f"{prefix}/{key}"] = value
    return output


def build_final_validation_metrics(
    airport_metrics_df: pd.DataFrame,
    best_postprocessing: dict[str, Any],
    cache_seconds: float,
    search_seconds: float,
    training_result: FoldTrainingResult,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "validation/objective": float(best_postprocessing["objective"]),
        "validation/macro_flight_balanced_accuracy": nanmean(
            airport_metrics_df["flight_balanced_accuracy"].tolist()
        ),
        "validation/macro_flight_specificity": nanmean(
            airport_metrics_df["flight_specificity"].tolist()
        ),
        "validation/macro_flight_sensitivity": nanmean(
            airport_metrics_df["flight_sensitivity"].tolist()
        ),
        "validation/macro_point_aucpr": nanmean(
            airport_metrics_df["point_aucpr"].tolist()
        ),
        "validation/macro_point_aucroc": nanmean(
            airport_metrics_df["point_aucroc"].tolist()
        ),
        "validation/best_core_probability_threshold": float(
            best_postprocessing["core_probability_threshold"]
        ),
        "validation/best_core_max_interruption_seconds": int(
            best_postprocessing["core_max_interruption_seconds"]
        ),
        "validation/min_core_segment_duration_seconds": (
            MIN_CORE_SEGMENT_DURATION_SECONDS
        ),
        "validation/postprocessing_cache_build_seconds": cache_seconds,
        "validation/postprocessing_search_seconds": search_seconds,
        "training/best_epoch": training_result.best_epoch,
        "training/best_val_aucpr": training_result.best_val_aucpr,
        "training/final_val_loss": training_result.final_val_metrics["loss"],
        "training/final_val_aucpr": training_result.final_val_metrics["aucpr"],
        "training/final_val_aucroc": training_result.final_val_metrics["aucroc"],
        "training/class_weight_0": training_result.class_weight_0,
        "training/class_weight_1": training_result.class_weight_1,
    }

    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "")
        for metric_name in ("precision", "recall", "f1"):
            column = f"segment_{metric_name}_iou{suffix}"
            metrics[
                f"validation/macro_segment_{metric_name}_iou{suffix}"
            ] = nanmean(airport_metrics_df[column].tolist())

    return metrics


def run_final_training(args: argparse.Namespace) -> None:
    seed_everything(RANDOM_STATE)

    source_run_path = normalize_wandb_run_path(args.source_run)
    source_entity, source_project, source_run_id = source_run_path.split("/")
    source_run, model_config = fetch_source_run_config(source_run_path)

    project = args.project or source_project
    entity = args.entity or source_entity
    output_dir = Path(args.output_dir) / f"source_{source_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_objective = source_run.summary.get("oof/objective")
    except Exception:
        source_objective = None

    repro = reproducibility_metadata()
    with (output_dir / "reproducibility.json").open("w", encoding="utf-8") as file:
        json.dump(json_compatible(repro), file, indent=2)

    initial_config = {
        **model_config,
        "source_run_path": source_run_path,
        "source_run_url": getattr(source_run, "url", args.source_run),
        "source_sweep_objective": source_objective,
        "airports": AIRPORTS,
        "validation_size": VAL_SIZE,
        "random_state": RANDOM_STATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "core_probability_thresholds": CORE_PROBABILITY_THRESHOLDS,
        "core_max_interruption_seconds": CORE_MAX_INTERRUPTION_SECONDS,
        "min_core_segment_duration_seconds": MIN_CORE_SEGMENT_DURATION_SECONDS,
        "postprocessing_workers": POSTPROCESSING_WORKERS,
        "reproducibility": repro,
    }

    run_name = args.run_name or f"final-model-from-{source_run_id}"
    with wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        job_type="final-all-airports-training",
        group="pointwise-tcn-final-model",
        tags=[
            "pointwise-tcn",
            "final-model",
            "all-airports",
            "core-only",
            "deterministic",
            "amp",
        ],
        config=json_compatible(initial_config),
        mode=args.wandb_mode,
    ) as run:
        print(f"Source run: {source_run_path}", flush=True)
        print(
            "Selected model configuration:\n"
            + json.dumps(json_compatible(model_config), indent=2),
            flush=True,
        )
        print(f"Device: {DEVICE}", flush=True)
        print(f"AMP enabled: {USE_AMP}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)

        sequences_by_airport: dict[str, list[SequenceRecord]] = {}
        feature_names: list[str] | None = None

        for airport, path in AIRPORT_FILES.items():
            sequences, current_feature_names = build_sequences_for_airport(
                airport=airport,
                data_path=Path(path),
                feature_variant=model_config["feature_variant"],
                track_windows=model_config["track_windows"],
            )
            sequences_by_airport[airport] = sequences
            if feature_names is None:
                feature_names = current_feature_names
            elif feature_names != current_feature_names:
                raise RuntimeError("Feature definitions differ between airports")

        if feature_names is None:
            raise RuntimeError("No feature names were produced")

        (
            split_indices,
            train_raw_by_airport,
            validation_raw_by_airport,
            train_raw,
            validation_raw,
        ) = build_final_splits(sequences_by_airport)

        split_df = build_split_dataframe(
            train_raw_by_airport,
            validation_raw_by_airport,
        )
        split_df.to_parquet(
            output_dir / "train_validation_split.parquet",
            index=False,
        )
        with (output_dir / "split_indices.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(json_compatible(split_indices), file, indent=2)

        scaler = fit_scaler(train_raw)
        joblib.dump(scaler, output_dir / "final_scaler.pkl")

        train_sequences = transform_sequences(train_raw, scaler)
        validation_sequences = transform_sequences(validation_raw, scaler)
        validation_scaled_by_airport = {
            airport: transform_sequences(
                validation_raw_by_airport[airport],
                scaler,
            )
            for airport in AIRPORTS
        }

        parameter_probe = PointwiseTCN(
            n_features=len(feature_names),
            filters=TCN_FILTERS,
            kernel_size=TCN_KERNEL_SIZE,
            dilations=TCN_DILATIONS,
            dense_units=DENSE_UNITS,
            dropout=model_config["dropout"],
        )
        trainable_parameters = count_trainable_parameters(parameter_probe)
        del parameter_probe
        receptive_field = (
            1
            + 2 * (TCN_KERNEL_SIZE - 1) * sum(TCN_DILATIONS)
        )

        run.config.update(
            {
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "n_train_flights": len(train_raw),
                "n_validation_flights": len(validation_raw),
                "trainable_parameters": trainable_parameters,
                "receptive_field": receptive_field,
                "tcn_filters": TCN_FILTERS,
                "tcn_kernel_size": TCN_KERNEL_SIZE,
                "tcn_dilations": TCN_DILATIONS,
                "dense_units": DENSE_UNITS,
            },
            allow_val_change=True,
        )

        training_result = train_final_model(
            train_sequences=train_sequences,
            validation_sequences=validation_sequences,
            n_features=len(feature_names),
            learning_rate=model_config["learning_rate"],
            dropout=model_config["dropout"],
            positive_weight_scale=model_config["positive_weight_scale"],
            run=run,
        )
        training_result.history.to_parquet(
            output_dir / "training_history.parquet",
            index=False,
        )

        probabilities_by_airport: dict[str, list[np.ndarray]] = {}
        for airport_index, airport in enumerate(AIRPORTS):
            probabilities_by_airport[airport] = predict_probabilities(
                training_result.model,
                validation_scaled_by_airport[airport],
                loader_seed=RANDOM_STATE + 1000 + airport_index,
            )

        cache_dir = output_dir / "_postprocessing_cache"
        cache_start = time.perf_counter()
        manifest = build_postprocessing_cache(
            cache_dir=cache_dir,
            sequences_by_airport=validation_raw_by_airport,
            probabilities_by_airport=probabilities_by_airport,
        )
        cache_seconds = float(time.perf_counter() - cache_start)

        search_results, best_postprocessing, search_seconds = (
            optimize_postprocessing(manifest)
        )
        search_results.to_parquet(
            output_dir / "core_postprocessing_search_results.parquet",
            index=False,
        )
        with (output_dir / "best_core_postprocessing.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                json_compatible(best_postprocessing),
                file,
                indent=2,
            )

        shutil.rmtree(cache_dir, ignore_errors=True)

        validation_dir = output_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        airport_metric_records: list[dict[str, Any]] = []
        all_flight_results: list[pd.DataFrame] = []
        all_boundary_results: list[pd.DataFrame] = []

        for airport in AIRPORTS:
            airport_dir = validation_dir / airport
            flight_results, boundary_results, metrics = (
                evaluate_best_postprocessing_for_airport(
                    airport=airport,
                    sequences=validation_raw_by_airport[airport],
                    probabilities=probabilities_by_airport[airport],
                    best_postprocessing=best_postprocessing,
                    fold_dir=airport_dir,
                )
            )

            old_prediction_path = airport_dir / "oof_predictions.parquet"
            new_prediction_path = airport_dir / "validation_predictions.parquet"
            if old_prediction_path.exists():
                os.replace(old_prediction_path, new_prediction_path)

            flight_results.to_parquet(
                airport_dir / "validation_flight_results.parquet",
                index=False,
            )
            if not boundary_results.empty:
                boundary_results.to_parquet(
                    airport_dir / "validation_boundary_matches.parquet",
                    index=False,
                )

            airport_metric_records.append(metrics)
            all_flight_results.append(flight_results)
            if not boundary_results.empty:
                all_boundary_results.append(boundary_results)

            run.log(numeric_metrics(f"validation/{airport}", metrics))

        airport_metrics_df = pd.DataFrame(airport_metric_records)
        airport_metrics_df.to_parquet(
            output_dir / "validation_metrics_by_airport.parquet",
            index=False,
        )
        validation_flight_results = pd.concat(
            all_flight_results,
            ignore_index=True,
        )
        validation_flight_results.to_parquet(
            output_dir / "validation_flight_results.parquet",
            index=False,
        )
        if all_boundary_results:
            pd.concat(
                all_boundary_results,
                ignore_index=True,
            ).to_parquet(
                output_dir / "validation_boundary_matches.parquet",
                index=False,
            )

        final_validation_metrics = build_final_validation_metrics(
            airport_metrics_df=airport_metrics_df,
            best_postprocessing=best_postprocessing,
            cache_seconds=cache_seconds,
            search_seconds=search_seconds,
            training_result=training_result,
        )
        run.log(final_validation_metrics)
        for key, value in final_validation_metrics.items():
            run.summary[key] = value

        checkpoint_postprocessing = {
            "core_probability_threshold": float(
                best_postprocessing["core_probability_threshold"]
            ),
            "core_max_interruption_seconds": int(
                best_postprocessing["core_max_interruption_seconds"]
            ),
            "min_core_segment_duration_seconds": (
                MIN_CORE_SEGMENT_DURATION_SECONDS
            ),
            "smoothing": None,
            "edge_expansion": None,
        }

        checkpoint = {
            "model_state_dict": cpu_state_dict(training_result.model),
            "model_config": {
                "n_features": len(feature_names),
                "filters": TCN_FILTERS,
                "kernel_size": TCN_KERNEL_SIZE,
                "dilations": TCN_DILATIONS,
                "dense_units": DENSE_UNITS,
                "dropout": model_config["dropout"],
            },
            "feature_names": feature_names,
            "feature_setup": model_config["feature_setup"],
            "feature_variant": model_config["feature_variant"],
            "track_window_preset": model_config["track_window_preset"],
            "sampled_track_window_preset": model_config[
                "sampled_track_window_preset"
            ],
            "effective_track_window_preset": model_config[
                "effective_track_window_preset"
            ],
            "track_windows": model_config["track_windows"],
            "learning_rate": model_config["learning_rate"],
            "positive_weight_scale": model_config["positive_weight_scale"],
            "label_col": LABEL_COL,
            "airports": AIRPORTS,
            "source_run_path": source_run_path,
            "source_sweep_id": "dd911dyu",
            "best_epoch": training_result.best_epoch,
            "best_val_aucpr": training_result.best_val_aucpr,
            "final_val_metrics": training_result.final_val_metrics,
            "class_weight_0": training_result.class_weight_0,
            "class_weight_1": training_result.class_weight_1,
            "validation_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
            "reproducibility": repro,
            "postprocessing": checkpoint_postprocessing,
            "validation_objective": float(best_postprocessing["objective"]),
        }
        torch.save(checkpoint, output_dir / "final_model.pt")

        complete_config = {
            **initial_config,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "trainable_parameters": trainable_parameters,
            "receptive_field": receptive_field,
            "n_train_flights": len(train_raw),
            "n_validation_flights": len(validation_raw),
            "best_epoch": training_result.best_epoch,
            "class_weight_0": training_result.class_weight_0,
            "class_weight_1": training_result.class_weight_1,
            "best_core_postprocessing": best_postprocessing,
            "validation_metrics": final_validation_metrics,
        }
        with (output_dir / "final_config.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(json_compatible(complete_config), file, indent=2)

        run.log(
            {
                "validation/metrics_by_airport_table": wandb.Table(
                    dataframe=airport_metrics_df
                ),
                "validation/core_postprocessing_search_table": wandb.Table(
                    dataframe=search_results
                ),
            }
        )

        flight_table = validation_flight_results
        if len(flight_table) > 5000:
            flight_table = flight_table.sample(
                n=5000,
                random_state=RANDOM_STATE,
            )
        run.log(
            {
                "validation/flight_results_table": wandb.Table(
                    dataframe=flight_table
                )
            }
        )

        artifact = wandb.Artifact(
            name=f"final-all-airports-{run.id}",
            type="final-model-evaluation-bundle",
            metadata=json_compatible(
                {
                    "source_run_path": source_run_path,
                    "source_sweep_id": "dd911dyu",
                    "best_epoch": training_result.best_epoch,
                    "validation_objective": best_postprocessing["objective"],
                    "best_core_postprocessing": checkpoint_postprocessing,
                    "feature_setup": model_config["feature_setup"],
                    "feature_variant": model_config["feature_variant"],
                    "sampled_track_window_preset": model_config[
                        "sampled_track_window_preset"
                    ],
                    "effective_track_window_preset": model_config[
                        "effective_track_window_preset"
                    ],
                    "track_windows": model_config["track_windows"],
                }
            ),
        )
        artifact.add_dir(local_path=str(output_dir))
        run.log_artifact(artifact)

        print("\nFinal training complete.", flush=True)
        print(
            "Best validation core-postprocessing configuration:\n"
            + json.dumps(json_compatible(best_postprocessing), indent=2),
            flush=True,
        )
        print(
            f"Cache build: {cache_seconds / 60:.2f} min; "
            f"search: {search_seconds / 60:.2f} min",
            flush=True,
        )
        print(f"Artifacts saved to: {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        default=DEFAULT_SOURCE_RUN,
        help=(
            "W&B run URL or entity/project/run_id whose model "
            "hyperparameters should be used."
        ),
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT"),
        help="Destination W&B project; defaults to the source project.",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="Destination W&B entity; defaults to the source entity.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--run-name",
        default=None,
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_final_training(parse_args())
