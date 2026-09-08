from __future__ import annotations

import numpy as np

from .utils import safe_divide

def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]

def segment_iou(
    first: tuple[int, int],
    second: tuple[int, int],
) -> float:
    first_start, first_end = first
    second_start, second_end = second
    intersection = max(
        0,
        min(first_end, second_end) - max(first_start, second_start) + 1,
    )
    first_length = first_end - first_start + 1
    second_length = second_end - second_start + 1
    union = first_length + second_length - intersection
    return safe_divide(intersection, union)

def match_segments(
    ground_truth_segments: list[tuple[int, int]],
    predicted_segments: list[tuple[int, int]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], int, int, int]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for gt_index, gt_segment in enumerate(ground_truth_segments):
        for pred_index, pred_segment in enumerate(predicted_segments):
            iou = segment_iou(gt_segment, pred_segment)
            candidate_pairs.append((iou, gt_index, pred_index))

    candidate_pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, gt_index, pred_index in candidate_pairs:
        if iou < iou_threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, float(iou)))

    tp = len(matches)
    fp = len(predicted_segments) - tp
    fn = len(ground_truth_segments) - tp
    return matches, tp, fp, fn
