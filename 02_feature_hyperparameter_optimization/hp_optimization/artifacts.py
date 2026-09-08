from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .config import (
    AIRPORTS, DENSE_UNITS, LABEL_COL, MIN_CORE_SEGMENT_DURATION_SECONDS,
    TCN_DILATIONS, TCN_FILTERS, TCN_KERNEL_SIZE,
)
from .training import FoldTrainingResult
from .utils import json_compatible

def save_fold_training_artifacts(
    fold_dir: Path,
    test_airport: str,
    training_result: FoldTrainingResult,
    scaler: StandardScaler,
    split_df: pd.DataFrame,
    feature_names: list[str],
    run_config: dict[str, Any],
) -> None:
    """Persist a completed fold immediately and release it from memory."""

    fold_dir.mkdir(parents=True, exist_ok=True)

    model_config = {
        "n_features": len(feature_names),
        "filters": TCN_FILTERS,
        "kernel_size": TCN_KERNEL_SIZE,
        "dilations": TCN_DILATIONS,
        "dense_units": DENSE_UNITS,
        "dropout": float(run_config["dropout"]),
    }
    checkpoint = {
        "model_state_dict": training_result.model.state_dict(),
        "model_config": model_config,
        "feature_names": feature_names,
        "feature_setup": run_config["feature_setup"],
        "feature_variant": run_config["feature_variant"],
        "sampled_track_window_preset": run_config["track_window_preset"],
        "effective_track_window_preset": run_config[
            "effective_track_window_preset"
        ],
        "track_windows": run_config["track_windows"],
        "label_col": LABEL_COL,
        "test_airport": test_airport,
        "train_airports": [
            airport for airport in AIRPORTS if airport != test_airport
        ],
        "best_epoch": training_result.best_epoch,
        "best_val_aucpr": training_result.best_val_aucpr,
        "final_val_metrics": training_result.final_val_metrics,
        "class_weight_0": training_result.class_weight_0,
        "class_weight_1": training_result.class_weight_1,
        "postprocessing": None,
    }

    torch.save(checkpoint, fold_dir / "model.pt")
    joblib.dump(scaler, fold_dir / "scaler.pkl")
    split_df.to_parquet(fold_dir / "flight_splits.parquet", index=False)
    training_result.history.to_parquet(
        fold_dir / "training_history.parquet",
        index=False,
    )

def attach_postprocessing_to_fold_artifacts(
    fold_dir: Path,
    best_postprocessing: dict[str, Any],
) -> None:
    """Add the selected global postprocessing config to a saved checkpoint."""

    postprocessing = {
        "core_probability_threshold": float(
            best_postprocessing["core_probability_threshold"]
        ),
        "core_max_interruption_seconds": int(
            best_postprocessing["core_max_interruption_seconds"]
        ),
        "min_core_segment_duration_seconds": MIN_CORE_SEGMENT_DURATION_SECONDS,
        "smoothing": None,
        "edge_expansion": None,
    }

    checkpoint_path = fold_dir / "model.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["postprocessing"] = postprocessing

    temporary_path = fold_dir / "model.pt.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)

    with (fold_dir / "best_core_postprocessing.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(json_compatible(postprocessing), file, indent=2)
