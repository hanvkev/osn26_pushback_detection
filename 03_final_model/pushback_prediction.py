"""Final-model pushback detection using the trained pointwise TCN.

The public ``predict_pushback()`` function accepts a traffic ``Flight`` and an
airport and returns the predicted pushback segment as a ``Flight`` or ``None``.

A process-local LRU cache loads the PyTorch model and scaler only once per
process.
"""

from __future__ import annotations

import multiprocessing as mp
import os

# Required by CUDA for deterministic cuBLAS operations. Set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from traffic.core.flight import Flight
from traffic.core.structure import Airport


DeviceName = Literal["cpu", "cuda", "auto"]

MODULE_ROOT = Path(__file__).resolve().parent
MODEL_DIR = MODULE_ROOT / "model"

DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "TRAFFIC_PUSHBACK_MODEL_PATH",
        MODEL_DIR / "final_model.pt",
    )
)

DEFAULT_SCALER_PATH = Path(
    os.environ.get(
        "TRAFFIC_PUSHBACK_SCALER_PATH",
        MODEL_DIR / "final_scaler.pkl",
    )
)


# =============================================================================
# Postprocessing semantics
# =============================================================================

# Core threshold, allowed interruption and minimum duration are loaded from
# final_model.pt exactly as optimized.
#
# After the best core candidate has been selected, its boundaries are expanded
# using the fixed edge configuration below.
#
# Training and validation use nominally 1 Hz trajectories.
NOMINAL_SAMPLE_INTERVAL_SECONDS = 1.0

EDGE_PROBABILITY_THRESHOLD = 0.5
EDGE_GAP_SECONDS = 30


# =============================================================================
# Exact model architecture used during training
# =============================================================================


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation_rate) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            filters,
            kernel_size,
            dilation=dilation_rate,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            filters,
            filters,
            kernel_size,
            dilation=dilation_rate,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.shortcut = (
            nn.Conv1d(in_channels, filters, kernel_size=1)
            if in_channels != filters
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        return self.relu(x + shortcut)


class PointwiseTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        filters: int,
        kernel_size: int,
        dilations: list[int],
        dense_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = n_features

        for dilation_rate in dilations:
            blocks.append(
                ResidualTCNBlock(
                    in_channels=in_channels,
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    dropout=dropout,
                )
            )
            in_channels = filters

        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(filters, dense_units, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dense_units, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return self.head(x).squeeze(1)


# =============================================================================
# Metadata and feature engineering
# =============================================================================


@dataclass(frozen=True)
class PipelineMetadata:
    feature_variant: str
    feature_names: tuple[str, ...]
    track_windows: tuple[int, ...]
    gs_threshold: float
    core_probability_threshold: float
    core_max_interruption_seconds: int
    min_core_segment_duration_seconds: int


def compute_gs_diff(gs: np.ndarray, gs_threshold: float) -> np.ndarray:
    diff = np.diff(gs, prepend=gs[0]).astype(np.float32)
    diff[gs < gs_threshold] = 0.0
    return diff


def compute_track_diff(
    gs: np.ndarray,
    track: np.ndarray,
    gs_threshold: float,
) -> np.ndarray:
    previous = np.roll(track, 1)
    difference = ((track - previous + 180.0) % 360.0) - 180.0
    difference = np.abs(difference).astype(np.float32)
    difference[0] = 0.0
    difference[gs < gs_threshold] = 0.0
    return difference


def compute_track_change(
    gs: np.ndarray,
    track: np.ndarray,
    n: int,
    gs_threshold: float,
) -> np.ndarray:
    length = len(gs)
    track_change = np.zeros(length, dtype=np.float32)
    moving_idx = np.where(gs > gs_threshold)[0]

    if len(moving_idx) < 2:
        return track_change

    n_moving = len(moving_idx)
    track_rad = np.radians(track[moving_idx])
    sin_track = np.sin(track_rad)
    cos_track = np.cos(track_rad)

    cumulative_sin = np.concatenate([[0.0], np.cumsum(sin_track)])
    cumulative_cos = np.concatenate([[0.0], np.cumsum(cos_track)])

    positions = np.searchsorted(moving_idx, np.arange(length))
    positions = np.clip(positions, 0, n_moving)

    before_start = np.maximum(0, positions - n)
    before_end = positions
    after_start = positions
    after_end = np.minimum(n_moving, positions + n)

    before_count = before_end - before_start
    after_count = after_end - after_start
    valid = (before_count > 0) & (after_count > 0)

    before_count_safe = np.maximum(1, before_count)
    after_count_safe = np.maximum(1, after_count)

    sin_before = (
        cumulative_sin[before_end] - cumulative_sin[before_start]
    ) / before_count_safe
    cos_before = (
        cumulative_cos[before_end] - cumulative_cos[before_start]
    ) / before_count_safe
    sin_after = (
        cumulative_sin[after_end] - cumulative_sin[after_start]
    ) / after_count_safe
    cos_after = (
        cumulative_cos[after_end] - cumulative_cos[after_start]
    ) / after_count_safe

    angle_before = np.degrees(np.arctan2(sin_before, cos_before)) % 360.0
    angle_after = np.degrees(np.arctan2(sin_after, cos_after)) % 360.0

    difference = ((angle_after - angle_before + 180.0) % 360.0) - 180.0
    track_change = np.abs(difference).astype(np.float32)
    track_change[~valid] = 0.0
    return track_change


def _resolve_feature_flags(feature_variant: str) -> tuple[bool, bool, bool]:
    valid_variants = {
        "gs_trackdiff",
        "gs_gsdiff_trackdiff",
        "gs_circular",
        "gs_gsdiff_circular",
        "gs_trackdiff_circular",
        "gs_gsdiff_trackdiff_circular",
    }
    if feature_variant not in valid_variants:
        raise ValueError(f"Unknown feature_variant: {feature_variant}")

    return (
        "gsdiff" in feature_variant,
        "trackdiff" in feature_variant,
        "circular" in feature_variant,
    )


def _build_feature_matrix(
    gs: np.ndarray,
    track: np.ndarray,
    metadata: PipelineMetadata,
) -> np.ndarray:
    use_gs_diff, use_track_diff, use_circular = _resolve_feature_flags(
        metadata.feature_variant
    )

    features: list[np.ndarray] = [gs.astype(np.float32)]
    feature_names = ["compute_gs"]

    if use_gs_diff:
        features.append(compute_gs_diff(gs, metadata.gs_threshold))
        feature_names.append("compute_gs_diff")

    if use_track_diff:
        features.append(
            compute_track_diff(gs, track, metadata.gs_threshold)
        )
        feature_names.append("compute_track_diff")

    if use_circular:
        for window in metadata.track_windows:
            features.append(
                compute_track_change(
                    gs=gs,
                    track=track,
                    n=int(window),
                    gs_threshold=metadata.gs_threshold,
                )
            )
            feature_names.append(f"circular_track_change_{window}")

    if tuple(feature_names) != metadata.feature_names:
        raise RuntimeError(
            "Generated feature order does not match final_model.pt: "
            f"generated={feature_names}, "
            f"expected={list(metadata.feature_names)}"
        )

    return np.column_stack(features).astype(np.float32)


# =============================================================================
# Process-local model/scaler runtime
# =============================================================================


class _PredictionRuntime:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        device_name: DeviceName,
        cpu_threads: int,
    ) -> None:
        self.device = _resolve_device(device_name)
        _configure_torch_runtime(self.device, cpu_threads)

        model_file = Path(model_path).expanduser().resolve()
        scaler_file = Path(scaler_path).expanduser().resolve()
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_file}")

        checkpoint = torch.load(
            model_file,
            map_location="cpu",
            weights_only=False,
        )
        model_config = checkpoint["model_config"]
        postprocessing = checkpoint["postprocessing"]

        self.metadata = PipelineMetadata(
            feature_variant=str(checkpoint["feature_variant"]),
            feature_names=tuple(
                str(v) for v in checkpoint["feature_names"]
            ),
            track_windows=tuple(
                int(v) for v in checkpoint["track_windows"]
            ),
            gs_threshold=float(
                checkpoint.get("gs_threshold", 0.5)
            ),
            core_probability_threshold=float(
                postprocessing["core_probability_threshold"]
            ),
            core_max_interruption_seconds=int(
                postprocessing["core_max_interruption_seconds"]
            ),
            min_core_segment_duration_seconds=int(
                postprocessing.get(
                    "min_core_segment_duration_seconds",
                    0,
                )
            ),
        )

        self.model = PointwiseTCN(
            n_features=int(model_config["n_features"]),
            filters=int(model_config["filters"]),
            kernel_size=int(model_config["kernel_size"]),
            dilations=[
                int(v) for v in model_config["dilations"]
            ],
            dense_units=int(model_config["dense_units"]),
            dropout=float(model_config["dropout"]),
        )
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.scaler = joblib.load(scaler_file)
        scaler_features = getattr(
            self.scaler,
            "n_features_in_",
            None,
        )
        expected_features = int(model_config["n_features"])

        if (
            scaler_features is not None
            and int(scaler_features) != expected_features
        ):
            raise RuntimeError(
                "Scaler/model feature mismatch: "
                f"scaler={scaler_features}, "
                f"model={expected_features}"
            )

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        scaled = self.scaler.transform(
            features
        ).astype(np.float32)

        tensor = torch.from_numpy(
            scaled[np.newaxis, :, :]
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)[0]

        return (
            probabilities
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


@lru_cache(maxsize=None)
def _get_runtime(
    model_path: str,
    scaler_path: str,
    device_name: DeviceName,
    cpu_threads: int,
) -> _PredictionRuntime:
    """Load one model/scaler pair per Python process and reuse it."""

    return _PredictionRuntime(
        model_path=model_path,
        scaler_path=scaler_path,
        device_name=device_name,
        cpu_threads=cpu_threads,
    )


def clear_prediction_runtime_cache() -> None:
    """Release cached model/scaler instances in the current process."""

    _get_runtime.cache_clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(
    device_name: DeviceName,
) -> torch.device:
    in_worker_process = (
        mp.current_process().name != "MainProcess"
    )

    if (
        in_worker_process
        and device_name != "cpu"
    ):
        raise RuntimeError(
            "Multiprocessing worker processes must use device='cpu'. "
            "For GPU inference, run prediction in the main process."
        )

    if device_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' requested, but CUDA is unavailable"
            )

        return torch.device("cuda")

    if device_name == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "device must be 'cpu', 'cuda', or 'auto'"
    )


def _configure_torch_runtime(
    device: torch.device,
    cpu_threads: int,
) -> None:
    cpu_threads = max(
        1,
        int(cpu_threads),
    )

    try:
        torch.set_num_threads(cpu_threads)
    except RuntimeError:
        pass

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.use_deterministic_algorithms(True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        try:
            torch.set_float32_matmul_precision(
                "highest"
            )
        except Exception:
            pass


# =============================================================================
# Final pushback prediction
# =============================================================================


class PredictionBasedPushback:
    """Detect pushback with the final pointwise PyTorch TCN.

    ``apply()`` returns only the selected pushback segment as a ``Flight`` or
    ``None``.

    Parameters
    ----------
    airport:
        ICAO airport code or an Airport object.
    model_path, scaler_path:
        Final model and scaler produced by the all-airports training.
    device:
        ``"cpu"`` by default. Use ``"cuda"`` only with a single-flight call or
        ``Traffic.eval(max_workers=1)``.
    cpu_threads:
        Threads used by one CPU worker. Keep at 1 when using
        ``Traffic.eval(max_workers=...)``.
    """

    def __init__(
        self,
        airport: str | Airport,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        device: DeviceName = "cpu",
        cpu_threads: int = 1,
    ) -> None:
        from traffic.data import airports

        self.airport = (
            airports[airport.upper()]
            if isinstance(airport, str)
            else airport
        )

        if self.airport is None:
            raise RuntimeError(
                "Airport information missing"
            )

        self.model_path = str(
            Path(model_path)
            .expanduser()
            .resolve()
        )
        self.scaler_path = str(
            Path(scaler_path)
            .expanduser()
            .resolve()
        )
        self.device = device
        self.cpu_threads = max(
            1,
            int(cpu_threads),
        )

    def apply(
        self,
        flight: Flight,
    ) -> Flight | None:
        runtime = _get_runtime(
            self.model_path,
            self.scaler_path,
            self.device,
            self.cpu_threads,
        )

        prepared = self._prepare_flight(
            flight,
            runtime.metadata,
        )

        if prepared is None:
            return None

        dataframe, valid_indices, features = prepared

        if (
            features is None
            or valid_indices.size == 0
        ):
            return None

        probabilities = runtime.predict(
            features
        )

        predicted = self._postprocess(
            dataframe=dataframe,
            valid_indices=valid_indices,
            probabilities=probabilities,
            metadata=runtime.metadata,
        )

        segment_data = predicted.loc[
            predicted["pushback"]
        ].copy()

        if segment_data.empty:
            return None

        return Flight(
            segment_data.reset_index(
                drop=True
            )
        )

    def _prepare_flight(
        self,
        flight: Flight,
        metadata: PipelineMetadata,
    ) -> (
        tuple[
            pd.DataFrame,
            np.ndarray,
            np.ndarray | None,
        ]
        | None
    ):
        required = [
            "timestamp",
            "latitude",
            "longitude",
            "geoaltitude",
        ]

        missing = [
            column
            for column in required
            if column not in flight.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required input columns: {missing}"
            )

        working_flight = Flight(
            flight.data.copy(
                deep=True
            )
        )

        altitude_limit = (
            float(self.airport.altitude)
            + 500.0
        )

        clipped_flight = (
            working_flight.clip(
                self.airport.shape
            )
        )

        if clipped_flight is None:
            return None

        low_points = clipped_flight.data[
            (
                clipped_flight.data[
                    "geoaltitude"
                ]
                < altitude_limit
            )
            | clipped_flight.data[
                "geoaltitude"
            ].isna()
        ]

        if low_points.empty:
            return None

        last_low_time = low_points[
            "timestamp"
        ].max()

        processed_flight = (
            working_flight.before(
                last_low_time,
                strict=False,
            )
        )

        if processed_flight is None:
            return None

        processed_flight = (
            processed_flight
            .cumulative_distance()
        )

        if processed_flight is None:
            return None

        processed_flight = (
            processed_flight.unwrap(
                "compute_track"
            )
        )

        if processed_flight is None:
            return None

        dataframe = (
            processed_flight.data
            .copy()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        dataframe["timestamp"] = (
            pd.to_datetime(
                dataframe["timestamp"]
            )
        )

        required_features = [
            "compute_gs",
            "compute_track",
        ]

        missing_features = [
            column
            for column in required_features
            if column
            not in dataframe.columns
        ]

        if missing_features:
            raise ValueError(
                "Missing feature columns after cumulative_distance(): "
                f"{missing_features}"
            )

        dataframe[
            "predicted_probability"
        ] = np.nan

        dataframe[
            "pushback"
        ] = False

        valid_mask = (
            dataframe[
                "compute_gs"
            ].notna()
            & dataframe[
                "compute_track"
            ].notna()
        )

        valid_indices = (
            dataframe.index[
                valid_mask
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        if valid_indices.size == 0:
            return (
                dataframe,
                valid_indices,
                None,
            )

        valid_df = dataframe.loc[
            valid_indices
        ]

        gs = valid_df[
            "compute_gs"
        ].to_numpy(
            dtype=np.float32
        )

        track = valid_df[
            "compute_track"
        ].to_numpy(
            dtype=np.float32
        )

        features = _build_feature_matrix(
            gs,
            track,
            metadata,
        )

        return (
            dataframe,
            valid_indices,
            features,
        )

    @staticmethod
    def _postprocess(
        dataframe: pd.DataFrame,
        valid_indices: np.ndarray,
        probabilities: np.ndarray,
        metadata: PipelineMetadata,
    ) -> pd.DataFrame:
        """Apply optimized core detection followed by fixed edge expansion.

        Stage 1 reproduces the optimized core postprocessing stored in
        ``final_model.pt``:

        1. Threshold pointwise probabilities with
           ``core_probability_threshold``.
        2. Join positive observations while the duration of the interruption
           between them is no larger than
           ``core_max_interruption_seconds``.
        3. Discard candidates shorter than
           ``min_core_segment_duration_seconds``.
        4. If several candidates remain, select exactly one by descending
           mean probability, duration and maximum probability. Stable sorting
           leaves the earliest candidate first when all three values tie.

        Stage 2 freezes the selected core candidate and expands only its
        boundaries using:

        - edge probability threshold: 0.5
        - edge gap: 30 seconds

        The edge stage cannot create or select another core candidate.
        """
        dataframe = dataframe.copy()

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if (
            len(probabilities)
            != len(valid_indices)
        ):
            raise RuntimeError(
                "Prediction length mismatch: "
                f"{len(probabilities)} probabilities for "
                f"{len(valid_indices)} points"
            )

        dataframe.loc[
            valid_indices,
            "predicted_probability",
        ] = probabilities

        dataframe["pushback"] = False

        valid_timestamps = (
            pd.to_datetime(
                dataframe.loc[
                    valid_indices,
                    "timestamp",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        # =====================================================================
        # Stage 1: optimized core detection and selection
        # =====================================================================

        positive_indices = (
            np.flatnonzero(
                probabilities
                >= metadata.core_probability_threshold
            )
        )

        if positive_indices.size == 0:
            return dataframe

        candidate_records: list[
            dict[str, float | int]
        ] = []

        candidate_start = 0

        for position in range(
            1,
            len(positive_indices) + 1,
        ):
            is_end = (
                position
                == len(positive_indices)
            )

            if not is_end:
                previous_index = int(
                    positive_indices[
                        position - 1
                    ]
                )

                current_index = int(
                    positive_indices[
                        position
                    ]
                )

                elapsed_seconds = float(
                    (
                        valid_timestamps.iloc[
                            current_index
                        ]
                        - valid_timestamps.iloc[
                            previous_index
                        ]
                    ).total_seconds()
                )

                interruption_seconds = max(
                    0.0,
                    elapsed_seconds
                    - NOMINAL_SAMPLE_INTERVAL_SECONDS,
                )

                is_end = (
                    interruption_seconds
                    > metadata.core_max_interruption_seconds
                )

            if is_end:
                indices = (
                    positive_indices[
                        candidate_start:position
                    ]
                )

                start_index = int(
                    indices[0]
                )

                end_index = int(
                    indices[-1]
                )

                duration_seconds = float(
                    (
                        valid_timestamps.iloc[
                            end_index
                        ]
                        - valid_timestamps.iloc[
                            start_index
                        ]
                    ).total_seconds()
                )

                if (
                    duration_seconds
                    >= metadata.min_core_segment_duration_seconds
                ):
                    candidate_records.append(
                        {
                            "start_index": start_index,
                            "end_index": end_index,
                            "duration_seconds": (
                                duration_seconds
                            ),
                            "probability_mean": float(
                                np.mean(
                                    probabilities[
                                        indices
                                    ]
                                )
                            ),
                            "probability_max": float(
                                np.max(
                                    probabilities[
                                        indices
                                    ]
                                )
                            ),
                        }
                    )

                candidate_start = position

        if not candidate_records:
            return dataframe

        candidates = (
            pd.DataFrame(
                candidate_records
            )
            .sort_values(
                [
                    "probability_mean",
                    "duration_seconds",
                    "probability_max",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                kind="mergesort",
            )
        )

        best = candidates.iloc[0]

        core_start_index = int(
            best["start_index"]
        )

        core_end_index = int(
            best["end_index"]
        )

        # =====================================================================
        # Stage 2: fixed edge expansion
        # =====================================================================

        edge_indices = np.flatnonzero(
            probabilities
            >= EDGE_PROBABILITY_THRESHOLD
        )

        if edge_indices.size == 0:
            raise RuntimeError(
                "No edge-positive points exist although "
                "a core candidate was selected."
            )

        left_position = int(
            np.searchsorted(
                edge_indices,
                core_start_index,
                side="left",
            )
        )

        right_position = int(
            np.searchsorted(
                edge_indices,
                core_end_index,
                side="right",
            )
            - 1
        )

        # Since the edge threshold is lower than the core threshold, the
        # selected core boundaries must also be edge-positive.
        if (
            left_position
            >= len(edge_indices)
            or int(
                edge_indices[
                    left_position
                ]
            )
            != core_start_index
            or right_position < 0
            or int(
                edge_indices[
                    right_position
                ]
            )
            != core_end_index
        ):
            raise RuntimeError(
                "Selected core boundaries are not edge-positive. "
                "EDGE_PROBABILITY_THRESHOLD must be less than or equal "
                "to the optimized core_probability_threshold."
            )

        # ---------------------------------------------------------------------
        # Expand left
        # ---------------------------------------------------------------------

        while left_position > 0:
            current_index = int(
                edge_indices[
                    left_position
                ]
            )

            previous_index = int(
                edge_indices[
                    left_position - 1
                ]
            )

            elapsed_seconds = float(
                (
                    valid_timestamps.iloc[
                        current_index
                    ]
                    - valid_timestamps.iloc[
                        previous_index
                    ]
                ).total_seconds()
            )

            interruption_seconds = max(
                0.0,
                elapsed_seconds
                - NOMINAL_SAMPLE_INTERVAL_SECONDS,
            )

            if (
                interruption_seconds
                > EDGE_GAP_SECONDS
            ):
                break

            left_position -= 1

        # ---------------------------------------------------------------------
        # Expand right
        # ---------------------------------------------------------------------

        while (
            right_position + 1
            < len(edge_indices)
        ):
            current_index = int(
                edge_indices[
                    right_position
                ]
            )

            next_index = int(
                edge_indices[
                    right_position + 1
                ]
            )

            elapsed_seconds = float(
                (
                    valid_timestamps.iloc[
                        next_index
                    ]
                    - valid_timestamps.iloc[
                        current_index
                    ]
                ).total_seconds()
            )

            interruption_seconds = max(
                0.0,
                elapsed_seconds
                - NOMINAL_SAMPLE_INTERVAL_SECONDS,
            )

            if (
                interruption_seconds
                > EDGE_GAP_SECONDS
            ):
                break

            right_position += 1

        final_start_index = int(
            edge_indices[
                left_position
            ]
        )

        final_end_index = int(
            edge_indices[
                right_position
            ]
        )

        final_start_time = (
            valid_timestamps.iloc[
                final_start_index
            ]
        )

        final_end_time = (
            valid_timestamps.iloc[
                final_end_index
            ]
        )

        # Return a continuous Flight segment between the expanded boundaries.
        # This also retains any original rows between two valid model points.
        selected_mask = (
            (
                dataframe["timestamp"]
                >= final_start_time
            )
            & (
                dataframe["timestamp"]
                <= final_end_time
            )
        )

        dataframe.loc[
            selected_mask,
            "pushback",
        ] = True

        dataframe["pushback"] = (
            dataframe["pushback"]
            .fillna(False)
            .astype(bool)
        )

        return dataframe


def predict_pushback(
    flight: Flight,
    airport: str | Airport,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    device: DeviceName = "cpu",
    cpu_threads: int = 1,
) -> Flight | None:
    """Return the predicted pushback segment for one flight.

    Parameters
    ----------
    flight:
        Input traffic ``Flight``.
    airport:
        ICAO airport code or an ``Airport`` object.
    model_path, scaler_path:
        Final model and scaler produced by the all-airports training.
    device:
        ``"cpu"`` by default. Use ``"cuda"`` for GPU inference in the main
        process.
    cpu_threads:
        Number of PyTorch CPU threads used for inference.

    Returns
    -------
    Flight | None
        Predicted pushback segment, or ``None`` when no pushback is detected.
    """

    return PredictionBasedPushback(
        airport=airport,
        model_path=model_path,
        scaler_path=scaler_path,
        device=device,
        cpu_threads=cpu_threads,
    ).apply(flight)


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SCALER_PATH",
    "NOMINAL_SAMPLE_INTERVAL_SECONDS",
    "EDGE_PROBABILITY_THRESHOLD",
    "EDGE_GAP_SECONDS",
    "PredictionBasedPushback",
    "predict_pushback",
    "clear_prediction_runtime_cache",
]