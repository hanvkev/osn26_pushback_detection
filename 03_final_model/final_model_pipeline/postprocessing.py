from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AIRPORTS, CORE_MAX_INTERRUPTION_SECONDS, CORE_PROBABILITY_THRESHOLDS,
    MIN_CORE_SEGMENT_DURATION_SECONDS, NOMINAL_SAMPLE_INTERVAL_SECONDS,
    OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT, OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT,
    OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT, RANDOM_STATE,
)
from .data import SequenceRecord
from .metrics import mask_to_segments
from .runtime import POSTPROCESSING_WORKERS
from .utils import f1_from_counts, json_compatible, nanmean, safe_divide

def postprocess_probabilities(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    probability_threshold: float,
    max_interruption_seconds: int,
    min_segment_duration_seconds: int,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Convert pointwise probabilities into at most one pushback-core segment.

    ``max_interruption_seconds`` is the maximum duration of non-core observations
    allowed between two positive core observations. With the 1 Hz input used in
    this study, a value of 0 joins directly consecutive positive observations but
    does not bridge a negative observation.
    """
    probabilities = np.asarray(probabilities, dtype=np.float32)
    positive_indices = np.flatnonzero(probabilities >= probability_threshold)
    predicted_mask = np.zeros(len(probabilities), dtype=bool)

    if len(positive_indices) == 0:
        return predicted_mask, None

    timestamp_series = pd.to_datetime(timestamps)
    candidate_records: list[dict[str, Any]] = []
    candidate_start = 0

    for position in range(1, len(positive_indices) + 1):
        is_end = position == len(positive_indices)
        if not is_end:
            previous_index = positive_indices[position - 1]
            current_index = positive_indices[position]
            elapsed_seconds = (
                timestamp_series[current_index] - timestamp_series[previous_index]
            ).total_seconds()
            interruption_seconds = max(
                0.0,
                elapsed_seconds - NOMINAL_SAMPLE_INTERVAL_SECONDS,
            )
            is_end = interruption_seconds > max_interruption_seconds

        if is_end:
            indices = positive_indices[candidate_start:position]
            start_index = int(indices[0])
            end_index = int(indices[-1])
            duration_seconds = float(
                (timestamp_series[end_index] - timestamp_series[start_index])
                .total_seconds()
            )

            if duration_seconds >= min_segment_duration_seconds:
                candidate_records.append(
                    {
                        "start_index": start_index,
                        "end_index": end_index,
                        "segment_start": timestamp_series[start_index],
                        "segment_end": timestamp_series[end_index],
                        "duration_seconds": duration_seconds,
                        "probability_mean": float(np.mean(probabilities[indices])),
                        "probability_max": float(np.max(probabilities[indices])),
                        "number_of_positive_points": int(len(indices)),
                    }
                )
            candidate_start = position

    if not candidate_records:
        return predicted_mask, None

    candidates = pd.DataFrame(candidate_records).sort_values(
        ["probability_mean", "duration_seconds", "probability_max"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    best = candidates.iloc[0].to_dict()
    start_index = int(best["start_index"])
    end_index = int(best["end_index"])
    predicted_mask[start_index : end_index + 1] = True
    return predicted_mask, best

def build_postprocessing_cache(
    cache_dir: Path,
    sequences_by_airport: dict[str, list[SequenceRecord]],
    probabilities_by_airport: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    """Write compact read-only arrays shared by CPU core-search workers."""

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ordered_sequences: list[SequenceRecord] = []
    ordered_probabilities: list[np.ndarray] = []
    airport_indices: list[int] = []

    for airport_index, airport in enumerate(AIRPORTS):
        sequences = sequences_by_airport[airport]
        probabilities = probabilities_by_airport[airport]
        if len(sequences) != len(probabilities):
            raise RuntimeError(f"{airport}: OOF sequence/probability mismatch")
        ordered_sequences.extend(sequences)
        ordered_probabilities.extend(probabilities)
        airport_indices.extend([airport_index] * len(sequences))

    lengths = np.asarray([sequence.length for sequence in ordered_sequences], dtype=np.int64)
    point_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    point_offsets[0] = 0
    np.cumsum(lengths, out=point_offsets[1:])
    total_points = int(point_offsets[-1])

    gt_segments_per_flight = [mask_to_segments(sequence.y >= 0.5) for sequence in ordered_sequences]
    gt_counts = np.asarray([len(segments) for segments in gt_segments_per_flight], dtype=np.int64)
    gt_offsets = np.empty(len(gt_counts) + 1, dtype=np.int64)
    gt_offsets[0] = 0
    np.cumsum(gt_counts, out=gt_offsets[1:])
    total_gt_segments = int(gt_offsets[-1])

    paths = {
        "point_offsets": str(cache_dir / "point_offsets.npy"),
        "timestamps_ns": str(cache_dir / "timestamps_ns.npy"),
        "probabilities": str(cache_dir / "probabilities.npy"),
        "airport_indices": str(cache_dir / "airport_indices.npy"),
        "gt_offsets": str(cache_dir / "gt_offsets.npy"),
        "gt_starts": str(cache_dir / "gt_starts.npy"),
        "gt_ends": str(cache_dir / "gt_ends.npy"),
    }
    np.save(paths["point_offsets"], point_offsets)
    np.save(paths["airport_indices"], np.asarray(airport_indices, dtype=np.int16))
    np.save(paths["gt_offsets"], gt_offsets)

    timestamps_memmap = np.lib.format.open_memmap(paths["timestamps_ns"], mode="w+", dtype=np.int64, shape=(total_points,))
    probabilities_memmap = np.lib.format.open_memmap(paths["probabilities"], mode="w+", dtype=np.float32, shape=(total_points,))
    gt_starts_memmap = np.lib.format.open_memmap(paths["gt_starts"], mode="w+", dtype=np.int32, shape=(total_gt_segments,))
    gt_ends_memmap = np.lib.format.open_memmap(paths["gt_ends"], mode="w+", dtype=np.int32, shape=(total_gt_segments,))

    for flight_index, (sequence, probabilities, gt_segments) in enumerate(zip(ordered_sequences, ordered_probabilities, gt_segments_per_flight)):
        point_start = int(point_offsets[flight_index])
        point_end = int(point_offsets[flight_index + 1])
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if len(probabilities) != sequence.length:
            raise RuntimeError(f"{sequence.airport}/{sequence.flight_id}: probability length mismatch")
        timestamps_memmap[point_start:point_end] = np.asarray(sequence.timestamps).astype("datetime64[ns]").astype(np.int64)
        probabilities_memmap[point_start:point_end] = probabilities

        gt_start = int(gt_offsets[flight_index])
        gt_end = int(gt_offsets[flight_index + 1])
        if gt_end > gt_start:
            gt_starts_memmap[gt_start:gt_end] = np.asarray([segment[0] for segment in gt_segments], dtype=np.int32)
            gt_ends_memmap[gt_start:gt_end] = np.asarray([segment[1] for segment in gt_segments], dtype=np.int32)

    for memmap in (timestamps_memmap, probabilities_memmap, gt_starts_memmap, gt_ends_memmap):
        memmap.flush()
    del timestamps_memmap, probabilities_memmap, gt_starts_memmap, gt_ends_memmap

    manifest = {
        "cache_dir": str(cache_dir),
        "airports": AIRPORTS,
        "n_flights": len(ordered_sequences),
        "total_points": total_points,
        "paths": paths,
    }
    with (cache_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(json_compatible(manifest), file, indent=2)
    return manifest

def select_best_candidate_fast(
    timestamps_ns: np.ndarray,
    probabilities: np.ndarray,
    positive_indices: np.ndarray,
    max_interruption_seconds: int,
) -> tuple[int, int] | None:
    """Select the same candidate as ``postprocess_probabilities``."""
    if positive_indices.size == 0:
        return None

    if positive_indices.size == 1:
        split_positions = np.empty(0, dtype=np.int64)
    else:
        elapsed_seconds = np.diff(timestamps_ns[positive_indices]) / 1_000_000_000.0
        interruption_seconds = np.maximum(0.0, elapsed_seconds - NOMINAL_SAMPLE_INTERVAL_SECONDS)
        split_positions = np.flatnonzero(interruption_seconds > max_interruption_seconds) + 1

    group_starts = np.concatenate(([0], split_positions))
    group_ends = np.concatenate((split_positions, [len(positive_indices)]))
    best_key: tuple[float, int, float] | None = None
    best_segment: tuple[int, int] | None = None

    for group_start, group_end in zip(group_starts, group_ends):
        indices = positive_indices[group_start:group_end]
        start_index = int(indices[0])
        end_index = int(indices[-1])
        values = probabilities[indices]
        key = (
            float(np.mean(values)),
            int(timestamps_ns[end_index] - timestamps_ns[start_index]),
            float(np.max(values)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_segment = (start_index, end_index)
    return best_segment

def _postprocessing_worker_init() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def _evaluate_postprocessing_task(
    manifest: dict[str, Any],
    probability_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate all core-interruption values for one probability threshold."""
    paths = manifest["paths"]
    point_offsets = np.load(paths["point_offsets"], mmap_mode="r")
    timestamps_ns = np.load(paths["timestamps_ns"], mmap_mode="r")
    probabilities_all = np.load(paths["probabilities"], mmap_mode="r")
    airport_indices = np.load(paths["airport_indices"], mmap_mode="r")
    gt_offsets = np.load(paths["gt_offsets"], mmap_mode="r")
    gt_starts = np.load(paths["gt_starts"], mmap_mode="r")
    gt_ends = np.load(paths["gt_ends"], mmap_mode="r")

    n_airports = len(manifest["airports"])
    n_gaps = len(CORE_MAX_INTERRUPTION_SECONDS)
    count_names = (
        "flight_tp", "flight_tn", "flight_fp", "flight_fn",
        "segment_tp_iou05", "segment_fp_iou05", "segment_fn_iou05",
        "segment_tp_iou075", "segment_fp_iou075", "segment_fn_iou075",
    )
    counts = {name: np.zeros((n_airports, n_gaps), dtype=np.int64) for name in count_names}

    for flight_index in range(int(manifest["n_flights"])):
        point_start = int(point_offsets[flight_index])
        point_end = int(point_offsets[flight_index + 1])
        timestamps = timestamps_ns[point_start:point_end]
        probabilities = probabilities_all[point_start:point_end]
        positive_indices = np.flatnonzero(probabilities >= float(probability_threshold))

        airport_index = int(airport_indices[flight_index])
        gt_start = int(gt_offsets[flight_index])
        gt_end = int(gt_offsets[flight_index + 1])
        current_gt_starts = gt_starts[gt_start:gt_end]
        current_gt_ends = gt_ends[gt_start:gt_end]
        n_gt_segments = int(gt_end - gt_start)
        gt_has_pushback = n_gt_segments > 0

        for gap_index, max_interruption_seconds in enumerate(CORE_MAX_INTERRUPTION_SECONDS):
            predicted_segment = select_best_candidate_fast(
                timestamps_ns=timestamps,
                probabilities=probabilities,
                positive_indices=positive_indices,
                max_interruption_seconds=max_interruption_seconds,
            )
            pred_has_pushback = predicted_segment is not None
            counts["flight_tp"][airport_index, gap_index] += int(gt_has_pushback and pred_has_pushback)
            counts["flight_tn"][airport_index, gap_index] += int(not gt_has_pushback and not pred_has_pushback)
            counts["flight_fp"][airport_index, gap_index] += int(not gt_has_pushback and pred_has_pushback)
            counts["flight_fn"][airport_index, gap_index] += int(gt_has_pushback and not pred_has_pushback)

            if predicted_segment is None:
                counts["segment_fn_iou05"][airport_index, gap_index] += n_gt_segments
                counts["segment_fn_iou075"][airport_index, gap_index] += n_gt_segments
                continue

            pred_start, pred_end = predicted_segment
            if n_gt_segments == 0:
                counts["segment_fp_iou05"][airport_index, gap_index] += 1
                counts["segment_fp_iou075"][airport_index, gap_index] += 1
                continue

            intersections = np.maximum(0, np.minimum(current_gt_ends, pred_end) - np.maximum(current_gt_starts, pred_start) + 1)
            gt_lengths = current_gt_ends - current_gt_starts + 1
            pred_length = pred_end - pred_start + 1
            unions = gt_lengths + pred_length - intersections
            best_iou = float(np.max(intersections / unions))

            for suffix, iou_threshold in (("iou05", 0.5), ("iou075", 0.75)):
                matched = int(best_iou >= iou_threshold)
                counts[f"segment_tp_{suffix}"][airport_index, gap_index] += matched
                counts[f"segment_fp_{suffix}"][airport_index, gap_index] += 1 - matched
                counts[f"segment_fn_{suffix}"][airport_index, gap_index] += n_gt_segments - matched

    result_rows: list[dict[str, Any]] = []
    for gap_index, max_interruption_seconds in enumerate(CORE_MAX_INTERRUPTION_SECONDS):
        airport_balanced_accuracies: list[float] = []
        airport_f1_iou05: list[float] = []
        airport_f1_iou075: list[float] = []
        row: dict[str, Any] = {
            "core_probability_threshold": float(probability_threshold),
            "core_max_interruption_seconds": int(max_interruption_seconds),
            "min_core_segment_duration_seconds": MIN_CORE_SEGMENT_DURATION_SECONDS,
        }

        for airport_index, airport in enumerate(manifest["airports"]):
            flight_tp = int(counts["flight_tp"][airport_index, gap_index])
            flight_tn = int(counts["flight_tn"][airport_index, gap_index])
            flight_fp = int(counts["flight_fp"][airport_index, gap_index])
            flight_fn = int(counts["flight_fn"][airport_index, gap_index])
            sensitivity = safe_divide(flight_tp, flight_tp + flight_fn)
            specificity = safe_divide(flight_tn, flight_tn + flight_fp)
            balanced_accuracy = nanmean([sensitivity, specificity])
            _, _, f1_iou05 = f1_from_counts(
                int(counts["segment_tp_iou05"][airport_index, gap_index]),
                int(counts["segment_fp_iou05"][airport_index, gap_index]),
                int(counts["segment_fn_iou05"][airport_index, gap_index]),
            )
            _, _, f1_iou075 = f1_from_counts(
                int(counts["segment_tp_iou075"][airport_index, gap_index]),
                int(counts["segment_fp_iou075"][airport_index, gap_index]),
                int(counts["segment_fn_iou075"][airport_index, gap_index]),
            )
            airport_balanced_accuracies.append(balanced_accuracy)
            airport_f1_iou05.append(f1_iou05)
            airport_f1_iou075.append(f1_iou075)
            row[f"{airport}_flight_balanced_accuracy"] = balanced_accuracy
            row[f"{airport}_segment_f1_iou05"] = f1_iou05
            row[f"{airport}_segment_f1_iou075"] = f1_iou075

        macro_flight_balanced_accuracy = nanmean(airport_balanced_accuracies)
        macro_segment_f1_iou05 = nanmean(airport_f1_iou05)
        macro_segment_f1_iou075 = nanmean(airport_f1_iou075)
        objective = (
            OBJECTIVE_FLIGHT_BALANCED_ACCURACY_WEIGHT * macro_flight_balanced_accuracy
            + OBJECTIVE_SEGMENT_F1_IOU05_WEIGHT * macro_segment_f1_iou05
            + OBJECTIVE_SEGMENT_F1_IOU075_WEIGHT * macro_segment_f1_iou075
        )
        row.update({
            "macro_flight_balanced_accuracy": macro_flight_balanced_accuracy,
            "macro_segment_f1_iou05": macro_segment_f1_iou05,
            "macro_segment_f1_iou075": macro_segment_f1_iou075,
            "objective": objective,
        })
        result_rows.append(row)
    return result_rows

def optimize_postprocessing(
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], float]:
    """Evaluate the fixed global core grid and return the best configuration."""
    tasks = list(CORE_PROBABILITY_THRESHOLDS)
    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    print(
        f"Core search: {len(tasks)} worker tasks, "
        f"{len(tasks) * len(CORE_MAX_INTERRUPTION_SECONDS)} configurations, "
        f"workers={POSTPROCESSING_WORKERS}",
        flush=True,
    )

    context = mp.get_context("spawn")
    worker_environment = {
        "POSTPROCESSING_WORKER": "1", "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1", "TORCH_NUM_INTEROP_THREADS": "1",
        "PYTHONHASHSEED": str(RANDOM_STATE),
    }
    previous_environment = {key: os.environ.get(key) for key in worker_environment}
    os.environ.update(worker_environment)
    try:
        with ProcessPoolExecutor(
            max_workers=POSTPROCESSING_WORKERS,
            mp_context=context,
            initializer=_postprocessing_worker_init,
        ) as executor:
            futures = {
                executor.submit(_evaluate_postprocessing_task, manifest, threshold): threshold
                for threshold in tasks
            }
            completed = 0
            for future in as_completed(futures):
                threshold = futures[future]
                rows.extend(future.result())
                completed += 1
                print(
                    f"Core task {completed}/{len(tasks)} complete "
                    f"(threshold={threshold:.2f})",
                    flush=True,
                )
    finally:
        for key, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value

    elapsed_seconds = float(time.perf_counter() - start_time)
    results = pd.DataFrame(rows)
    threshold_order = {value: index for index, value in enumerate(CORE_PROBABILITY_THRESHOLDS)}
    gap_order = {value: index for index, value in enumerate(CORE_MAX_INTERRUPTION_SECONDS)}
    results["configuration_id"] = [
        threshold_order[float(threshold)] * len(CORE_MAX_INTERRUPTION_SECONDS)
        + gap_order[int(gap)]
        for threshold, gap in zip(
            results["core_probability_threshold"],
            results["core_max_interruption_seconds"],
        )
    ]
    results = results.sort_values(
        ["objective", "macro_flight_balanced_accuracy", "macro_segment_f1_iou075", "macro_segment_f1_iou05", "configuration_id"],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best = json_compatible(results.iloc[0].to_dict())
    return results, best, elapsed_seconds
