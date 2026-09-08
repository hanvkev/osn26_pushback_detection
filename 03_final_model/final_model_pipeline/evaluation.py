from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from .config import IOU_THRESHOLDS, MIN_CORE_SEGMENT_DURATION_SECONDS
from .data import SequenceRecord
from .metrics import mask_to_segments, match_segments, segment_iou
from .postprocessing import postprocess_probabilities
from .utils import f1_from_counts, nanmean, safe_divide

def evaluate_flight(
    sequence: SequenceRecord,
    probabilities: np.ndarray,
    probability_threshold: float,
    max_interruption_seconds: int,
    min_segment_duration_seconds: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predicted_mask, best_candidate = postprocess_probabilities(
        timestamps=sequence.timestamps,
        probabilities=probabilities,
        probability_threshold=probability_threshold,
        max_interruption_seconds=max_interruption_seconds,
        min_segment_duration_seconds=min_segment_duration_seconds,
    )

    gt_mask = sequence.y >= 0.5
    gt_segments = mask_to_segments(gt_mask)
    predicted_segments = mask_to_segments(predicted_mask)

    gt_has_pb = len(gt_segments) > 0
    pred_has_pb = len(predicted_segments) > 0

    flight_tp = int(gt_has_pb and pred_has_pb)
    flight_tn = int(not gt_has_pb and not pred_has_pb)
    flight_fp = int(not gt_has_pb and pred_has_pb)
    flight_fn = int(gt_has_pb and not pred_has_pb)

    result: dict[str, Any] = {
        "airport": sequence.airport,
        "flight_id": sequence.flight_id,
        "gt_has_pushback": int(gt_has_pb),
        "pred_has_pushback": int(pred_has_pb),
        "flight_tp": flight_tp,
        "flight_tn": flight_tn,
        "flight_fp": flight_fp,
        "flight_fn": flight_fn,
        "n_gt_segments": len(gt_segments),
        "n_pred_segments": len(predicted_segments),
        "best_iou": 0.0,
        "predicted_start": None,
        "predicted_end": None,
        "predicted_duration_seconds": None,
    }

    if best_candidate is not None:
        result["predicted_start"] = best_candidate["segment_start"]
        result["predicted_end"] = best_candidate["segment_end"]
        result["predicted_duration_seconds"] = best_candidate[
            "duration_seconds"
        ]

    if gt_segments and predicted_segments:
        result["best_iou"] = max(
            segment_iou(gt_segment, predicted_segment)
            for gt_segment in gt_segments
            for predicted_segment in predicted_segments
        )

    for threshold in IOU_THRESHOLDS:
        matches, tp, fp, fn = match_segments(
            gt_segments,
            predicted_segments,
            threshold,
        )
        suffix = str(threshold).replace(".", "")
        result[f"segment_tp_iou{suffix}"] = tp
        result[f"segment_fp_iou{suffix}"] = fp
        result[f"segment_fn_iou{suffix}"] = fn

    # Boundary errors use matches at the primary IoU threshold of 0.5.
    primary_matches, _, _, _ = match_segments(
        gt_segments,
        predicted_segments,
        0.5,
    )
    boundary_records: list[dict[str, Any]] = []
    timestamp_series = pd.to_datetime(sequence.timestamps)

    for gt_index, pred_index, iou in primary_matches:
        gt_start_idx, gt_end_idx = gt_segments[gt_index]
        pred_start_idx, pred_end_idx = predicted_segments[pred_index]

        gt_start = timestamp_series[gt_start_idx]
        gt_end = timestamp_series[gt_end_idx]
        pred_start = timestamp_series[pred_start_idx]
        pred_end = timestamp_series[pred_end_idx]

        gt_duration = (gt_end - gt_start).total_seconds()
        pred_duration = (pred_end - pred_start).total_seconds()
        start_error = (pred_start - gt_start).total_seconds()
        end_error = (pred_end - gt_end).total_seconds()
        duration_error = pred_duration - gt_duration

        boundary_records.append(
            {
                "airport": sequence.airport,
                "flight_id": sequence.flight_id,
                "iou": iou,
                "start_error_seconds": start_error,
                "abs_start_error_seconds": abs(start_error),
                "end_error_seconds": end_error,
                "abs_end_error_seconds": abs(end_error),
                "duration_error_seconds": duration_error,
                "abs_duration_error_seconds": abs(duration_error),
            }
        )

    point_df = pd.DataFrame(
        {
            "airport": sequence.airport,
            "flight_id": sequence.flight_id,
            "timestamp": pd.to_datetime(sequence.timestamps),
            "y_true": gt_mask.astype(np.int8),
            "probability": probabilities.astype(np.float32),
            "predicted_pushback": predicted_mask.astype(np.int8),
        }
    )

    return result, point_df, pd.DataFrame(boundary_records)

def aggregate_airport_metrics(
    airport: str,
    flight_results: pd.DataFrame,
    point_results: pd.DataFrame,
    boundary_results: pd.DataFrame,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"airport": airport}

    flight_tp = int(flight_results["flight_tp"].sum())
    flight_tn = int(flight_results["flight_tn"].sum())
    flight_fp = int(flight_results["flight_fp"].sum())
    flight_fn = int(flight_results["flight_fn"].sum())

    sensitivity = safe_divide(flight_tp, flight_tp + flight_fn)
    specificity = safe_divide(flight_tn, flight_tn + flight_fp)
    balanced_accuracy = nanmean([sensitivity, specificity])
    flight_accuracy = safe_divide(
        flight_tp + flight_tn,
        flight_tp + flight_tn + flight_fp + flight_fn,
    )

    metrics.update(
        {
            "n_flights": len(flight_results),
            "flight_tp": flight_tp,
            "flight_tn": flight_tn,
            "flight_fp": flight_fp,
            "flight_fn": flight_fn,
            "flight_sensitivity": sensitivity,
            "flight_specificity": specificity,
            "flight_balanced_accuracy": balanced_accuracy,
            "flight_accuracy": flight_accuracy,
        }
    )

    for threshold in IOU_THRESHOLDS:
        suffix = str(threshold).replace(".", "")
        tp = int(flight_results[f"segment_tp_iou{suffix}"].sum())
        fp = int(flight_results[f"segment_fp_iou{suffix}"].sum())
        fn = int(flight_results[f"segment_fn_iou{suffix}"].sum())
        precision, recall, f1 = f1_from_counts(tp, fp, fn)
        metrics.update(
            {
                f"segment_tp_iou{suffix}": tp,
                f"segment_fp_iou{suffix}": fp,
                f"segment_fn_iou{suffix}": fn,
                f"segment_precision_iou{suffix}": precision,
                f"segment_recall_iou{suffix}": recall,
                f"segment_f1_iou{suffix}": f1,
            }
        )

    y_true = point_results["y_true"].to_numpy()
    probability = point_results["probability"].to_numpy()
    point_prediction = point_results["predicted_pushback"].to_numpy()

    metrics["point_accuracy"] = float(accuracy_score(y_true, point_prediction))
    if len(np.unique(y_true)) >= 2:
        metrics["point_aucpr"] = float(average_precision_score(y_true, probability))
        metrics["point_aucroc"] = float(roc_auc_score(y_true, probability))
    else:
        metrics["point_aucpr"] = float("nan")
        metrics["point_aucroc"] = float("nan")

    metrics["mean_best_iou"] = float(flight_results["best_iou"].mean())
    metrics["median_best_iou"] = float(flight_results["best_iou"].median())

    if boundary_results.empty:
        for column in [
            "iou",
            "start_error_seconds",
            "abs_start_error_seconds",
            "end_error_seconds",
            "abs_end_error_seconds",
            "duration_error_seconds",
            "abs_duration_error_seconds",
        ]:
            metrics[f"matched_mean_{column}"] = float("nan")
            metrics[f"matched_median_{column}"] = float("nan")
    else:
        for column in [
            "iou",
            "start_error_seconds",
            "abs_start_error_seconds",
            "end_error_seconds",
            "abs_end_error_seconds",
            "duration_error_seconds",
            "abs_duration_error_seconds",
        ]:
            metrics[f"matched_mean_{column}"] = float(
                boundary_results[column].mean()
            )
            metrics[f"matched_median_{column}"] = float(
                boundary_results[column].median()
            )

    return metrics

def evaluate_best_postprocessing_for_airport(
    airport: str,
    sequences: list[SequenceRecord],
    probabilities: list[np.ndarray],
    best_postprocessing: dict[str, Any],
    fold_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the detailed evaluation once for the selected configuration."""

    fold_dir.mkdir(parents=True, exist_ok=True)
    flight_records: list[dict[str, Any]] = []
    boundary_frames: list[pd.DataFrame] = []
    timestamp_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []

    for sequence, probability in zip(sequences, probabilities):
        flight_record, point_df, boundary_df = evaluate_flight(
            sequence=sequence,
            probabilities=probability,
            probability_threshold=float(
                best_postprocessing["core_probability_threshold"]
            ),
            max_interruption_seconds=int(
                best_postprocessing["core_max_interruption_seconds"]
            ),
            min_segment_duration_seconds=MIN_CORE_SEGMENT_DURATION_SECONDS,
        )
        flight_records.append(flight_record)
        timestamp_parts.append(
            point_df["timestamp"].to_numpy(dtype="datetime64[ns]")
        )
        y_parts.append(point_df["y_true"].to_numpy(dtype=np.int8))
        probability_parts.append(
            point_df["probability"].to_numpy(dtype=np.float32)
        )
        prediction_parts.append(
            point_df["predicted_pushback"].to_numpy(dtype=np.int8)
        )
        if not boundary_df.empty:
            boundary_frames.append(boundary_df)

    lengths = np.asarray([sequence.length for sequence in sequences], dtype=np.int64)
    total_points = int(lengths.sum())
    airport_values = np.full(total_points, airport, dtype=object)
    flight_id_values = np.repeat(
        np.asarray(
            [sequence.flight_id for sequence in sequences],
            dtype=object,
        ),
        lengths,
    )

    point_results = pd.DataFrame(
        {
            "airport": airport_values,
            "flight_id": flight_id_values,
            "timestamp": np.concatenate(timestamp_parts),
            "y_true": np.concatenate(y_parts),
            "probability": np.concatenate(probability_parts),
            "predicted_pushback": np.concatenate(prediction_parts),
        }
    )
    point_results.to_parquet(
        fold_dir / "oof_predictions.parquet",
        index=False,
    )

    flight_results = pd.DataFrame(flight_records)
    boundary_results = (
        pd.concat(boundary_frames, ignore_index=True)
        if boundary_frames
        else pd.DataFrame()
    )
    metrics = aggregate_airport_metrics(
        airport=airport,
        flight_results=flight_results,
        point_results=point_results,
        boundary_results=boundary_results,
    )
    del point_results
    return flight_results, boundary_results, metrics
