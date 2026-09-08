from __future__ import annotations

import numpy as np

from .config import CIRCULAR_FEATURE_VARIANTS, GS_THRESHOLD, NON_CIRCULAR_FEATURE_VARIANTS, TRACK_WINDOW_PRESETS

def compute_gs_diff(gs: np.ndarray, gs_threshold: float) -> np.ndarray:
    diff = np.diff(gs, prepend=gs[0]).astype(np.float32)
    diff[gs < gs_threshold] = 0.0
    return diff

def compute_track_diff(
    gs: np.ndarray,
    track: np.ndarray,
    gs_threshold: float,
) -> np.ndarray:
    """Point-to-point absolute circular track difference in degrees."""

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
    gs_threshold: float = 0.5,
) -> np.ndarray:
    """Difference between circular means before and after each point."""

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

def resolve_feature_configuration(
    feature_variant: str,
    sampled_track_window_preset: str,
) -> tuple[str, str | None, list[int]]:
    """Resolve sampled W&B feature parameters into the effective feature setup.

    ``track_window_preset`` remains an independent W&B sweep parameter. For the
    two non-circular feature variants it is intentionally ignored, so these
    variants use no circular window regardless of which preset W&B sampled.

    Returns
    -------
    feature_setup:
        A unique human-readable identifier for the effective feature setup.
    effective_track_window_preset:
        ``None`` for non-circular variants, otherwise the sampled preset.
    track_windows:
        Effective circular-track windows used to build the model inputs.
    """
    if feature_variant in NON_CIRCULAR_FEATURE_VARIANTS:
        return feature_variant, None, []

    if feature_variant not in CIRCULAR_FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature_variant: {feature_variant}")
    if sampled_track_window_preset not in TRACK_WINDOW_PRESETS:
        raise ValueError(
            f"Unknown track-window preset: {sampled_track_window_preset}"
        )

    return (
        f"{feature_variant}__{sampled_track_window_preset}",
        sampled_track_window_preset,
        list(TRACK_WINDOW_PRESETS[sampled_track_window_preset]),
    )

def resolve_feature_flags(feature_variant: str) -> tuple[bool, bool, bool]:
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

    use_gs_diff = "gsdiff" in feature_variant
    use_track_diff = "trackdiff" in feature_variant
    use_circular = "circular" in feature_variant
    return use_gs_diff, use_track_diff, use_circular

def build_feature_matrix(
    gs: np.ndarray,
    track: np.ndarray,
    feature_variant: str,
    track_windows: list[int],
) -> tuple[np.ndarray, list[str]]:
    use_gs_diff, use_track_diff, use_circular = resolve_feature_flags(
        feature_variant
    )

    features: list[np.ndarray] = [gs.astype(np.float32)]
    feature_names = ["compute_gs"]

    if use_gs_diff:
        features.append(compute_gs_diff(gs, GS_THRESHOLD))
        feature_names.append("compute_gs_diff")

    if use_track_diff:
        features.append(compute_track_diff(gs, track, GS_THRESHOLD))
        feature_names.append("compute_track_diff")

    if use_circular:
        for window in track_windows:
            features.append(
                compute_track_change(
                    gs=gs,
                    track=track,
                    n=int(window),
                    gs_threshold=GS_THRESHOLD,
                )
            )
            feature_names.append(f"circular_track_change_{window}")

    X = np.column_stack(features).astype(np.float32)
    return X, feature_names
