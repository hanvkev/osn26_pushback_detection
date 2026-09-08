#!/usr/bin/env python3
"""Bayesian W&B LOAO sweep for pointwise TCN pushback detection.

This module contains the complete W&B trial orchestration.
The repository-level ``sweep.py`` is the thin executable entry point.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wandb

from .artifacts import attach_postprocessing_to_fold_artifacts, save_fold_training_artifacts
from .config import (
    AIRPORT_FILES, AIRPORTS, BATCH_SIZE, CORE_MAX_INTERRUPTION_SECONDS,
    CORE_PROBABILITY_THRESHOLDS, DEFAULT_RUN_CONFIG, DENSE_UNITS, EPOCHS,
    GS_THRESHOLD, IOU_THRESHOLDS, LABEL_COL, MIN_CORE_SEGMENT_DURATION_SECONDS,
    OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT, OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT,
    OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT, PROJECT_ROOT, RANDOM_STATE,
    TCN_DILATIONS, TCN_FILTERS, TCN_KERNEL_SIZE, VAL_SIZE,
)
from .data import (
    SequenceRecord, build_sequences_for_airport, build_split_dataframe, collect_fold_sequences,
    create_airport_splits, fit_scaler, transform_sequences,
)
from .evaluation import evaluate_best_postprocessing_for_airport
from .features import resolve_feature_configuration
from .model import PointwiseTCN, count_trainable_parameters
from .postprocessing import build_postprocessing_cache, optimize_postprocessing
from .runtime import DEVICE, POSTPROCESSING_WORKERS, reproducibility_metadata, seed_everything
from .training import predict_probabilities, train_fold
from .utils import json_compatible, nanmean

def run_trial(args: argparse.Namespace) -> None:
    seed_everything(RANDOM_STATE)

    with wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="loao-sweep-trial",
        group="prd-tcn-final-loao-bayesian",
        tags=["prd-tcn", "loao", "final-bayesian-sweep", "core-only", "deterministic"],
        config=DEFAULT_RUN_CONFIG,
        mode=args.wandb_mode,
    ) as run:
        config = dict(run.config)
        feature_variant = str(config["feature_variant"])
        sampled_track_window_preset = str(config["track_window_preset"])
        (
            feature_setup,
            effective_track_window_preset,
            track_windows,
        ) = resolve_feature_configuration(
            feature_variant=feature_variant,
            sampled_track_window_preset=sampled_track_window_preset,
        )

        # Keep W&B's sampled ``track_window_preset`` unchanged. This is important
        # for the Bayesian optimizer. The effective value is logged separately
        # because non-circular variants intentionally ignore the sampled window.
        config["feature_setup"] = feature_setup
        config["effective_track_window_preset"] = effective_track_window_preset
        config["track_windows"] = track_windows

        output_dir = Path(args.output_dir) / run.id
        output_dir.mkdir(parents=True, exist_ok=True)

        repro = reproducibility_metadata()
        with (output_dir / "reproducibility.json").open("w", encoding="utf-8") as file:
            json.dump(json_compatible(repro), file, indent=2)
        run.config.update({"reproducibility": json_compatible(repro)}, allow_val_change=True)

        print(f"W&B run: {run.url}", flush=True)
        print(f"Device: {DEVICE}", flush=True)
        print(f"Output: {output_dir}", flush=True)
        print(f"Configuration: {json.dumps(json_compatible(config), indent=2)}")

        sequences_by_airport: dict[str, list[SequenceRecord]] = {}
        feature_names: list[str] | None = None

        for airport, path in AIRPORT_FILES.items():
            sequences, current_names = build_sequences_for_airport(
                airport=airport,
                data_path=Path(path),
                feature_variant=feature_variant,
                track_windows=track_windows,
            )
            sequences_by_airport[airport] = sequences
            if feature_names is None:
                feature_names = current_names
            elif feature_names != current_names:
                raise RuntimeError("Feature definitions differ between airports")

        if feature_names is None:
            raise RuntimeError("No feature names available")

        airport_splits = create_airport_splits(sequences_by_airport)

        receptive_field = 1 + 2 * (TCN_KERNEL_SIZE - 1) * sum(TCN_DILATIONS)
        parameter_probe = PointwiseTCN(
            n_features=len(feature_names),
            filters=TCN_FILTERS,
            kernel_size=TCN_KERNEL_SIZE,
            dilations=TCN_DILATIONS,
            dense_units=DENSE_UNITS,
            dropout=float(config["dropout"]),
        )
        trainable_parameters = count_trainable_parameters(parameter_probe)
        del parameter_probe

        run.config.update(
            {
                "feature_setup": feature_setup,
                "feature_variant": feature_variant,
                "sampled_track_window_preset": sampled_track_window_preset,
                "effective_track_window_preset": effective_track_window_preset,
                "track_windows": track_windows,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "airports": AIRPORTS,
                "label_col": LABEL_COL,
                "gs_threshold": GS_THRESHOLD,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "validation_size": VAL_SIZE,
                "random_state": RANDOM_STATE,
                "tcn_filters": TCN_FILTERS,
                "tcn_kernel_size": TCN_KERNEL_SIZE,
                "tcn_dilations": TCN_DILATIONS,
                "dense_units": DENSE_UNITS,
                "receptive_field": receptive_field,
                "trainable_parameters": trainable_parameters,
                "objective_flight_balanced_accuracy_weight": (
                    OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT
                ),
                "objective_segment_f1_iou05_weight": (
                    OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT
                ),
                "objective_segment_f1_iou075_weight": (
                    OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT
                ),
                "core_probability_thresholds": (
                    CORE_PROBABILITY_THRESHOLDS
                ),
                "core_max_interruption_seconds": (
                    CORE_MAX_INTERRUPTION_SECONDS
                ),
                "min_core_segment_duration_seconds": (
                    MIN_CORE_SEGMENT_DURATION_SECONDS
                ),
                "core_search_workers": POSTPROCESSING_WORKERS,
            },
            allow_val_change=True,
        )

        run.summary["feature_names"] = feature_names
        run.summary["n_features"] = len(feature_names)
        run.summary["trainable_parameters"] = trainable_parameters
        run.summary["receptive_field"] = receptive_field

        fold_training_metadata: dict[str, dict[str, Any]] = {}
        oof_sequences_by_airport: dict[str, list[SequenceRecord]] = {}
        oof_probabilities_by_airport: dict[str, list[np.ndarray]] = {}

        for fold_index, test_airport in enumerate(AIRPORTS):
            print(f"\n=== Fold: leave out {test_airport} ===", flush=True)
            seed_everything(RANDOM_STATE + fold_index)

            train_raw, validation_raw, test_raw = collect_fold_sequences(
                sequences_by_airport,
                airport_splits,
                test_airport,
            )
            scaler = fit_scaler(train_raw)
            train_sequences = transform_sequences(train_raw, scaler)
            validation_sequences = transform_sequences(validation_raw, scaler)
            test_sequences = transform_sequences(test_raw, scaler)

            split_df = build_split_dataframe(
                train_raw,
                validation_raw,
                test_raw,
            )

            training_result = train_fold(
                test_airport=test_airport,
                train_sequences=train_sequences,
                validation_sequences=validation_sequences,
                n_features=len(feature_names),
                learning_rate=float(config["learning_rate"]),
                dropout=float(config["dropout"]),
                positive_weight_scale=float(config["positive_weight_scale"]),
                loader_seed=RANDOM_STATE + 100 * fold_index,
                run=run,
            )

            probabilities = predict_probabilities(
                training_result.model,
                test_sequences,
                loader_seed=RANDOM_STATE + 100 * fold_index + 2,
            )
            training_result.model.to("cpu")

            fold_dir = output_dir / f"leave_out_{test_airport}"
            save_fold_training_artifacts(
                fold_dir=fold_dir,
                test_airport=test_airport,
                training_result=training_result,
                scaler=scaler,
                split_df=split_df,
                feature_names=feature_names,
                run_config={**config, "track_windows": track_windows},
            )
            fold_training_metadata[test_airport] = {
                "best_epoch": training_result.best_epoch,
                "best_internal_val_aucpr": (
                    training_result.best_val_aucpr
                ),
                "final_internal_val_loss": (
                    training_result.final_val_metrics["loss"]
                ),
                "final_internal_val_aucpr": (
                    training_result.final_val_metrics["aucpr"]
                ),
                "final_internal_val_aucroc": (
                    training_result.final_val_metrics["aucroc"]
                ),
                "class_weight_0": training_result.class_weight_0,
                "class_weight_1": training_result.class_weight_1,
            }
            oof_sequences_by_airport[test_airport] = test_raw
            oof_probabilities_by_airport[test_airport] = probabilities

            del (
                training_result,
                scaler,
                split_df,
                train_sequences,
                validation_sequences,
                test_sequences,
                train_raw,
                validation_raw,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cache_dir = output_dir / "postprocessing_cache"
        cache_build_start = time.perf_counter()
        cache_manifest = build_postprocessing_cache(
            cache_dir=cache_dir,
            sequences_by_airport=oof_sequences_by_airport,
            probabilities_by_airport=oof_probabilities_by_airport,
        )
        cache_build_seconds = float(time.perf_counter() - cache_build_start)

        search_results, best_postprocessing, search_seconds = (
            optimize_postprocessing(cache_manifest)
        )
        search_results.to_parquet(
            output_dir / "core_search_results.parquet",
            index=False,
        )
        with (output_dir / "best_core_postprocessing.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(best_postprocessing, file, indent=2)

        print("\nBest global core-postprocessing configuration:")
        print(json.dumps(best_postprocessing, indent=2))
        print(
            f"Cache build: {cache_build_seconds / 60:.2f} min; "
            f"search: {search_seconds / 60:.2f} min",
            flush=True,
        )

        fold_metric_records: list[dict[str, Any]] = []
        all_flight_results: list[pd.DataFrame] = []
        all_boundary_results: list[pd.DataFrame] = []

        for test_airport in AIRPORTS:
            fold_dir = output_dir / f"leave_out_{test_airport}"
            flight_results, boundary_results, fold_metrics = (
                evaluate_best_postprocessing_for_airport(
                    airport=test_airport,
                    sequences=oof_sequences_by_airport[test_airport],
                    probabilities=oof_probabilities_by_airport[test_airport],
                    best_postprocessing=best_postprocessing,
                    fold_dir=fold_dir,
                )
            )
            fold_metrics.update(fold_training_metadata[test_airport])
            fold_metric_records.append(fold_metrics)

            flat_fold_log = {
                f"oof/{test_airport}/{key}": value
                for key, value in fold_metrics.items()
                if key != "airport"
                and isinstance(value, (int, float, np.number))
            }
            run.log(flat_fold_log)

            flight_results.to_parquet(
                fold_dir / "oof_flight_results.parquet",
                index=False,
            )
            if not boundary_results.empty:
                boundary_results.to_parquet(
                    fold_dir / "oof_boundary_matches.parquet",
                    index=False,
                )

            attach_postprocessing_to_fold_artifacts(
                fold_dir=fold_dir,
                best_postprocessing=best_postprocessing,
            )

            all_flight_results.append(flight_results)
            if not boundary_results.empty:
                all_boundary_results.append(boundary_results)

        fold_metrics_df = pd.DataFrame(fold_metric_records)
        flight_results_df = pd.concat(all_flight_results, ignore_index=True)
        boundary_results_df = (
            pd.concat(all_boundary_results, ignore_index=True)
            if all_boundary_results
            else pd.DataFrame()
        )

        macro_flight_balanced_accuracy = nanmean(
            fold_metrics_df["flight_balanced_accuracy"].tolist()
        )
        macro_segment_f1_iou05 = nanmean(
            fold_metrics_df["segment_f1_iou05"].tolist()
        )
        macro_segment_f1_iou075 = nanmean(
            fold_metrics_df["segment_f1_iou075"].tolist()
        )
        objective_flight_component = (
            OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT
            * macro_flight_balanced_accuracy
        )
        objective_segment_iou05_component = (
            OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT
            * macro_segment_f1_iou05
        )
        objective_segment_iou075_component = (
            OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT
            * macro_segment_f1_iou075
        )
        objective = (
            objective_flight_component
            + objective_segment_iou05_component
            + objective_segment_iou075_component
        )

        final_metrics = {
            "oof/objective": objective,
            "oof/macro_flight_balanced_accuracy": (
                macro_flight_balanced_accuracy
            ),
            "oof/macro_flight_specificity": nanmean(
                fold_metrics_df["flight_specificity"].tolist()
            ),
            "oof/macro_flight_sensitivity": nanmean(
                fold_metrics_df["flight_sensitivity"].tolist()
            ),
            "oof/macro_point_aucpr": nanmean(
                fold_metrics_df["point_aucpr"].tolist()
            ),
            "oof/macro_point_aucroc": nanmean(
                fold_metrics_df["point_aucroc"].tolist()
            ),
            "oof/macro_segment_f1_iou05": macro_segment_f1_iou05,
            "oof/macro_segment_f1_iou075": macro_segment_f1_iou075,
            "oof/objective_flight_component": objective_flight_component,
            "oof/objective_segment_iou05_component": (
                objective_segment_iou05_component
            ),
            "oof/objective_segment_iou075_component": (
                objective_segment_iou075_component
            ),
            "oof/best_core_probability_threshold": float(
                best_postprocessing["core_probability_threshold"]
            ),
            "oof/best_core_max_interruption_seconds": int(
                best_postprocessing["core_max_interruption_seconds"]
            ),
            "oof/min_core_segment_duration_seconds": (
                MIN_CORE_SEGMENT_DURATION_SECONDS
            ),
            "oof/postprocessing_cache_build_seconds": cache_build_seconds,
            "oof/postprocessing_search_seconds": search_seconds,
            "oof/total_flight_tp": int(flight_results_df["flight_tp"].sum()),
            "oof/total_flight_tn": int(flight_results_df["flight_tn"].sum()),
            "oof/total_flight_fp": int(flight_results_df["flight_fp"].sum()),
            "oof/total_flight_fn": int(flight_results_df["flight_fn"].sum()),
        }

        for threshold in IOU_THRESHOLDS:
            suffix = str(threshold).replace(".", "")
            for metric_name in ("precision", "recall", "f1"):
                column = f"segment_{metric_name}_iou{suffix}"
                final_metrics[
                    f"oof/macro_segment_{metric_name}_iou{suffix}"
                ] = nanmean(fold_metrics_df[column].tolist())

        run.log(final_metrics)
        for key, value in final_metrics.items():
            run.summary[key] = value

        fold_metrics_df.to_parquet(
            output_dir / "fold_metrics.parquet",
            index=False,
        )
        flight_results_df.to_parquet(
            output_dir / "oof_flight_results.parquet",
            index=False,
        )
        if not boundary_results_df.empty:
            boundary_results_df.to_parquet(
                output_dir / "oof_boundary_matches.parquet",
                index=False,
            )

        complete_config = {
            **config,
            "track_windows": track_windows,
            "feature_names": feature_names,
            "architecture": {
                "filters": TCN_FILTERS,
                "kernel_size": TCN_KERNEL_SIZE,
                "dilations": TCN_DILATIONS,
                "dense_units": DENSE_UNITS,
            },
            "best_core_postprocessing": {
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
            },
            "objective_definition": {
                "flight_balanced_accuracy_weight": (
                    OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT
                ),
                "segment_f1_iou05_weight": (
                    OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT
                ),
                "segment_f1_iou075_weight": (
                    OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT
                ),
            },
        }
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
            json.dump(json_compatible(complete_config), file, indent=2)

        keep_postprocessing_cache = (
            os.environ.get("KEEP_POSTPROCESSING_CACHE", "0") == "1"
        )
        if not keep_postprocessing_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        run.summary["postprocessing_cache_retained"] = int(
            keep_postprocessing_cache
        )

        run.log({"oof/fold_metrics_table": wandb.Table(dataframe=fold_metrics_df)})
        run.log(
            {
                "oof/postprocessing_search_table": wandb.Table(
                    dataframe=search_results.head(240)
                )
            }
        )
        flight_table_df = flight_results_df
        if len(flight_table_df) > 5000:
            flight_table_df = flight_table_df.sample(
                n=5000,
                random_state=RANDOM_STATE,
            )
        run.log(
            {
                "oof/flight_results_table": wandb.Table(
                    dataframe=flight_table_df
                )
            }
        )

        artifact = wandb.Artifact(
            name=f"loao-sweep-run-{run.id}",
            type="model-evaluation-bundle",
            metadata=json_compatible(
                {
                    "objective": objective,
                    "macro_flight_balanced_accuracy": (
                        macro_flight_balanced_accuracy
                    ),
                    "macro_segment_f1_iou05": macro_segment_f1_iou05,
                    "macro_segment_f1_iou075": macro_segment_f1_iou075,
                    "best_postprocessing": complete_config[
                        "best_core_postprocessing"
                    ],
                    "feature_setup": feature_setup,
                    "feature_variant": feature_variant,
                    "sampled_track_window_preset": sampled_track_window_preset,
                    "effective_track_window_preset": effective_track_window_preset,
                    "track_windows": track_windows,
                    "feature_names": feature_names,
                }
            ),
        )
        artifact.add_dir(local_path=str(output_dir))
        run.log_artifact(artifact)

        print("\nFinal OOF metrics:")
        for key, value in final_metrics.items():
            print(f"{key}: {value}")
        print(f"Artifacts saved to: {output_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "pushback-detection"),
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            PROJECT_ROOT
            / "outputs"
            / "hyperparameter_optimization"
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(
            "Ignoring W&B sweep CLI arguments; "
            "values are read from wandb.config:",
            unknown_args,
            flush=True,
        )

    return args
