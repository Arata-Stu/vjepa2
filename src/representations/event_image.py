from __future__ import annotations

import numpy as np


def _normalize_counts_with_percentile(counts: np.ndarray, percentile: float) -> np.ndarray:
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D, got shape={counts.shape}")
    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    if counts.size == 0 or not np.any(counts > 0):
        return counts.astype(np.float32, copy=False)

    positive = counts[counts > 0]
    threshold = float(np.percentile(positive, percentile)) if positive.size > 0 else float(counts.max())
    if threshold <= 0.0:
        threshold = float(counts.max())
    if threshold <= 0.0:
        return counts.astype(np.float32, copy=False)
    return (np.clip(counts, 0.0, threshold) / threshold).astype(np.float32, copy=False)


def accumulate_events_to_rgb(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    shape: tuple[int, int],
    *,
    percentile: float = 99.0,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert events into a red/blue event image over a white background.

    Positive events map to red, negative events map to blue. The returned tensor
    is CHW in the `[0, 1]` range, and the activity volume is a single-channel
    count map `[1, H, W]` before percentile normalization.
    """
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"shape must be positive, got {shape}")

    x_arr = np.asarray(x).reshape(-1)
    y_arr = np.asarray(y).reshape(-1)
    p_arr = np.asarray(p).reshape(-1)
    if not (x_arr.size == y_arr.size == p_arr.size):
        raise ValueError(
            "event arrays must have identical length, got "
            f"x={x_arr.size}, y={y_arr.size}, p={p_arr.size}"
        )

    x_int = np.floor(x_arr).astype(np.int64, copy=False)
    y_int = np.floor(y_arr).astype(np.int64, copy=False)
    pos_mask = p_arr > 0
    valid = (
        (x_int >= 0)
        & (x_int < width)
        & (y_int >= 0)
        & (y_int < height)
    )

    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    if np.any(valid):
        x_valid = x_int[valid]
        y_valid = y_int[valid]
        pos_valid = pos_mask[valid]
        if np.any(pos_valid):
            np.add.at(pos, (y_valid[pos_valid], x_valid[pos_valid]), 1.0)
        if np.any(~pos_valid):
            np.add.at(neg, (y_valid[~pos_valid], x_valid[~pos_valid]), 1.0)

    pos_norm = _normalize_counts_with_percentile(pos, percentile)
    neg_norm = _normalize_counts_with_percentile(neg, percentile)

    dominate_pos = pos_norm >= neg_norm
    intensity_pos = pos_norm * dominate_pos
    intensity_neg = neg_norm * (~dominate_pos)

    red = np.ones((height, width), dtype=np.float32)
    green = np.ones((height, width), dtype=np.float32)
    blue = np.ones((height, width), dtype=np.float32)
    green -= intensity_pos
    blue -= intensity_pos
    red -= intensity_neg
    green -= intensity_neg

    rgb_hwc = np.stack(
        [
            np.clip(red, 0.0, 1.0),
            np.clip(green, 0.0, 1.0),
            np.clip(blue, 0.0, 1.0),
        ],
        axis=-1,
    )
    rgb_chw = np.moveaxis(rgb_hwc, -1, 0).astype(dtype, copy=False)
    activity_volume = (pos + neg)[None, ...].astype(np.float32, copy=False)
    return rgb_chw, activity_volume
