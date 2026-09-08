from __future__ import annotations

import os
import numpy as np

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

FINAL_DATA_ROOT = (
    DATA_ROOT
    / "train_validation"
    / "05_final"
)

AIRPORT_FILES = {
    "LSZH": FINAL_DATA_ROOT / "LSZH" / "LSZH_V1.parquet",
    "ESSA": FINAL_DATA_ROOT / "ESSA" / "ESSA_V1.parquet",
    "RJTT": FINAL_DATA_ROOT / "RJTT" / "RJTT_V1.parquet",
    "YPPH": FINAL_DATA_ROOT / "YPPH" / "YPPH_V1.parquet",
}

AIRPORTS = list(AIRPORT_FILES)
LABEL_COL = "Pushback"

RANDOM_STATE = 42
VAL_SIZE = 0.15
GS_THRESHOLD = 0.5

EPOCHS = 200
BATCH_SIZE = 128
MIN_LR = 1e-5
EARLY_STOPPING_PATIENCE = 10
LR_REDUCE_PATIENCE = 4
LR_REDUCE_FACTOR = 0.5

TCN_FILTERS = 64
TCN_KERNEL_SIZE = 5
TCN_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 96]
DENSE_UNITS = 64

IOU_THRESHOLDS = (0.3, 0.5, 0.7, 0.75, 0.9, 0.95)

OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT = 0.50
OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT = 0.20
OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT = 0.30

CORE_PROBABILITY_THRESHOLDS = np.round(np.arange(0.35, 1.001, 0.01), 2).tolist()
CORE_MAX_INTERRUPTION_SECONDS = (0, 15, 30, 45, 60, 90)
MIN_CORE_SEGMENT_DURATION_SECONDS = 0
NOMINAL_SAMPLE_INTERVAL_SECONDS = 1.0

TRACK_WINDOW_PRESETS = {
    "single_5": [5],
    "single_15": [15],
    "single_30": [30],
    "dual_5_15": [5, 15],
    "baseline_10_30": [10, 30],
    "triple_5_15_30": [5, 15, 30],
    "triple_10_30_60": [10, 30, 60],
}
NON_CIRCULAR_FEATURE_VARIANTS = ("gs_trackdiff", "gs_gsdiff_trackdiff")
CIRCULAR_FEATURE_VARIANTS = (
    "gs_circular",
    "gs_gsdiff_circular",
    "gs_trackdiff_circular",
    "gs_gsdiff_trackdiff_circular",
)
FEATURE_VARIANTS = list(NON_CIRCULAR_FEATURE_VARIANTS) + list(CIRCULAR_FEATURE_VARIANTS)

DEFAULT_RUN_CONFIG = {
    "feature_variant": "gs_gsdiff_circular",
    "track_window_preset": "baseline_10_30",
    "learning_rate": 1e-3,
    "dropout": 0.2,
    "positive_weight_scale": 1.0,
}

# Resource configuration is intentionally environment-aware so SLURM settings
# can override local defaults without editing source files.
def resolve_resource_config(is_postprocessing_worker: bool) -> dict[str, int]:
    if is_postprocessing_worker:
        return {"n_cpus": 1, "n_interop": 1, "num_workers": 0, "postprocessing_workers": 1}
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return {
        "n_cpus": n_cpus,
        "n_interop": int(os.environ.get("TORCH_NUM_INTEROP_THREADS", 2)),
        "num_workers": min(4, n_cpus),
        "postprocessing_workers": max(
            1,
            min(int(os.environ.get("POSTPROCESSING_WORKERS", "8")), n_cpus, len(CORE_PROBABILITY_THRESHOLDS)),
        ),
    }
