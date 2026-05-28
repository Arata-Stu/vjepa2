from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont


try:
    _RESAMPLING = Image.Resampling
except AttributeError:  # Pillow < 9.1
    _RESAMPLING = Image


TaskTarget = Literal["semantic", "depth"]


@dataclass(frozen=True)
class LabelSourceInfo:
    target: TaskTarget
    source_kind: str
    label_dataset_path: str
    label_length: int
    embedded_dataset_name: str
    resolved_ts_source: str
    num_timestamps: int
    missing_reason: str


@dataclass(frozen=True)
class WindowLabelInfo:
    target: TaskTarget
    window_index: int
    source_window_index: int
    label_index: int
    label_index_mode: str
    label_shape: tuple[int, ...]
    valid_window: bool
    source_kind: str
    label_dataset_path: str
    missing_reason: str
    unique_classes: list[int]
    depth_min: float | None
    depth_max: float | None
    depth_p05: float | None
    depth_p95: float | None
    depth_valid_ratio: float | None


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_h5_string(value.item())
        if value.size == 1:
            return _decode_h5_string(value.reshape(-1)[0])
    return str(value)


def _safe_attr(attrs: h5py.AttributeManager, key: str, default: Any = None) -> Any:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_h5_attr_str(h5f: h5py.File, key: str, default: str = "") -> str:
    if key not in h5f.attrs:
        return default
    return _decode_h5_string(h5f.attrs[key]).strip()


def _load_h5_attr_int(h5f: h5py.File, key: str) -> int | None:
    if key not in h5f.attrs:
        return None
    value = h5f.attrs[key]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    try:
        return int(value)
    except Exception:
        return None


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            if "voxels" not in h5f:
                return False
            if h5f["voxels"].ndim != 4 or int(h5f["voxels"].shape[0]) <= 0:
                return False
            rep = _load_h5_attr_str(h5f, "representation", default="")
            return rep == "event_voxel_grid_m3ed"
    except Exception:
        return False


def _select_files(
    input_path: Path | None,
    dataset_root: Path | None,
    recursive: bool,
    file_pattern: str,
    max_files: int | None,
) -> list[Path]:
    if input_path is not None and dataset_root is not None:
        raise ValueError("use either --input_path or --dataset_root, not both")
    if input_path is None and dataset_root is None:
        raise ValueError("either --input_path or --dataset_root is required")

    if input_path is not None:
        files = [input_path]
    else:
        assert dataset_root is not None
        iterator = dataset_root.rglob(file_pattern) if recursive else dataset_root.glob(file_pattern)
        files = sorted(p for p in iterator if p.is_file())

    files = [p.resolve() for p in files if _is_voxel_h5(p)]
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _infer_split_polarity(h5f: h5py.File, channels: int) -> bool:
    attr = _safe_attr(h5f.attrs, "split_polarity", None)
    if attr is not None:
        return bool(int(attr))
    return channels % 2 == 0


def _pick_evenly_spaced(indices: np.ndarray, count: int) -> list[int]:
    if indices.size == 0 or count <= 0:
        return []
    if indices.size <= count:
        return [int(v) for v in indices.tolist()]
    picked = np.linspace(0, indices.size - 1, num=count, dtype=np.int64)
    return [int(indices[int(i)]) for i in picked.tolist()]


def _select_window_indices(
    *,
    n_samples: int,
    valid_mask: np.ndarray,
    num_samples: int,
    selection_mode: str,
    explicit_indices: list[int] | None,
) -> tuple[list[int], str]:
    all_indices = np.arange(n_samples, dtype=np.int64)
    if explicit_indices:
        selected = sorted(set(int(i) for i in explicit_indices if 0 <= int(i) < n_samples))
        return selected, "explicit"

    if selection_mode == "valid":
        eligible = all_indices[valid_mask > 0]
    elif selection_mode == "invalid":
        eligible = all_indices[valid_mask <= 0]
    elif selection_mode == "all":
        eligible = all_indices
    else:
        raise ValueError(f"unsupported selection_mode: {selection_mode}")

    if eligible.size > 0:
        return _pick_evenly_spaced(eligible, int(num_samples)), selection_mode

    fallback = _pick_evenly_spaced(all_indices, int(num_samples))
    return fallback, f"{selection_mode}_fallback_all"


def _vote_score_map(voxel: np.ndarray, *, split_polarity: bool, polarity_order: str) -> np.ndarray:
    if voxel.ndim != 3:
        raise ValueError(f"expected voxel [C,H,W], got shape={voxel.shape}")

    if split_polarity and int(voxel.shape[0]) % 2 == 0 and int(voxel.shape[0]) > 1:
        half = int(voxel.shape[0]) // 2
        if polarity_order == "negpos":
            neg_bins = np.abs(np.asarray(voxel[:half], dtype=np.float32))
            pos_bins = np.abs(np.asarray(voxel[half:], dtype=np.float32))
        elif polarity_order == "posneg":
            pos_bins = np.abs(np.asarray(voxel[:half], dtype=np.float32))
            neg_bins = np.abs(np.asarray(voxel[half:], dtype=np.float32))
        else:
            raise ValueError(f"unsupported polarity_order: {polarity_order}")
        return np.sign(pos_bins - neg_bins).sum(axis=0)

    return np.sign(np.asarray(voxel, dtype=np.float32)).sum(axis=0)


def _normalize_to_uint8(arr: np.ndarray, *, percentile: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    upper = float(np.percentile(arr, percentile))
    if upper <= 0:
        upper = float(arr.max()) if arr.size > 0 else 0.0
    if upper <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip(arr / upper, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _voxel_to_activity_rgb(voxel: np.ndarray) -> np.ndarray:
    activity = np.abs(np.asarray(voxel, dtype=np.float32)).sum(axis=0)
    gray = _normalize_to_uint8(activity, percentile=99.5)
    return np.stack([gray, gray, gray], axis=-1)


def _voxel_to_polarity_rgb(voxel: np.ndarray, *, split_polarity: bool, polarity_order: str) -> np.ndarray:
    vote_score = _vote_score_map(
        voxel=voxel,
        split_polarity=split_polarity,
        polarity_order=polarity_order,
    )
    intensity = _normalize_to_uint8(np.abs(vote_score), percentile=99.5)
    rgb = np.full((vote_score.shape[0], vote_score.shape[1], 3), 127, dtype=np.uint8)
    pos = vote_score > 0
    neg = vote_score < 0
    rgb[pos, 0] = intensity[pos]
    rgb[pos, 1] = np.minimum(255, intensity[pos] // 2 + 48)
    rgb[pos, 2] = 32
    rgb[neg, 0] = 32
    rgb[neg, 1] = np.minimum(255, intensity[neg] // 2 + 48)
    rgb[neg, 2] = intensity[neg]
    return rgb


def _semantic_palette() -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        (220, 20, 60),
        (255, 127, 80),
        (255, 215, 0),
        (50, 205, 50),
        (0, 170, 0),
        (70, 130, 180),
        (65, 105, 225),
        (138, 43, 226),
        (255, 105, 180),
        (255, 140, 0),
        (0, 206, 209),
        (205, 92, 92),
        (154, 205, 50),
        (123, 104, 238),
        (244, 164, 96),
        (46, 139, 87),
        (210, 180, 140),
        (30, 144, 255),
        (199, 21, 133),
        (188, 143, 143),
    ]
    for idx, color in enumerate(colors):
        palette[idx] = np.asarray(color, dtype=np.uint8)
    palette[255] = np.asarray((24, 24, 24), dtype=np.uint8)
    return palette


def _depth_palette_stops() -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([0.0, 0.2, 0.4, 0.65, 0.82, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            (13, 8, 135),
            (75, 3, 161),
            (125, 3, 168),
            (187, 55, 84),
            (249, 142, 8),
            (240, 249, 33),
        ],
        dtype=np.float32,
    )
    return positions, colors


def _squeeze_label_to_hw(arr: np.ndarray, *, target: TaskTarget) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"expected 2D/3D label map, got shape={arr.shape}")
    if arr.shape[0] == 1:
        return arr[0]
    if arr.shape[-1] == 1:
        return arr[..., 0]
    if target == "depth":
        raise ValueError(f"unsupported depth shape={arr.shape}; expected singleton-channel")
    raise ValueError(f"unsupported semantic shape={arr.shape}; expected class-index map")


def _semantic_to_rgb(label: np.ndarray, *, ignore_index: int) -> np.ndarray:
    palette = _semantic_palette()
    label = np.asarray(label, dtype=np.int64)
    rgb = palette[np.clip(label, 0, 255)]
    rgb[label == int(ignore_index)] = palette[int(ignore_index)]
    return rgb


def _colorize_depth(depth: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float | None]]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    rgb = np.full((depth.shape[0], depth.shape[1], 3), 24, dtype=np.uint8)
    if not np.any(valid):
        return rgb, {
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
            "valid_ratio": 0.0,
        }

    values = depth[valid]
    lo = float(np.percentile(values, 5.0))
    hi = float(np.percentile(values, 95.0))
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    pos, col = _depth_palette_stops()
    colorized = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
    for c in range(3):
        colorized[..., c] = np.interp(normalized, pos, col[:, c])
    rgb[valid] = np.clip(np.round(colorized[valid]), 0, 255).astype(np.uint8)
    return rgb, {
        "min": float(values.min()),
        "max": float(values.max()),
        "p05": lo,
        "p95": hi,
        "valid_ratio": float(np.mean(valid)),
    }


def _alpha_blend(base_rgb: np.ndarray, label_rgb: np.ndarray, valid_mask: np.ndarray, alpha: float) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32)
    label = np.asarray(label_rgb, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    out = base.copy()
    out[mask] = (1.0 - float(alpha)) * base[mask] + float(alpha) * label[mask]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _resize_semantic_label(label: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    arr = np.asarray(label, dtype=np.uint8)
    if arr.shape == (int(target_h), int(target_w)):
        return arr
    image = Image.fromarray(arr, mode="L")
    resized = image.resize((int(target_w), int(target_h)), resample=_RESAMPLING.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _resize_depth_map(depth: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.shape == (int(target_h), int(target_w)):
        return arr
    image = Image.fromarray(arr, mode="F")
    resized = image.resize((int(target_w), int(target_h)), resample=_RESAMPLING.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _resize_valid_mask(mask: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.shape == (int(target_h), int(target_w)):
        return arr.astype(bool)
    image = Image.fromarray(arr * 255, mode="L")
    resized = image.resize((int(target_w), int(target_h)), resample=_RESAMPLING.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0)


def _find_first_matching_dataset(
    h5f: h5py.File,
    *,
    candidates: tuple[str, ...],
    length0: int,
    min_ndim: int = 3,
) -> str | None:
    for path in candidates:
        if path not in h5f:
            continue
        ds = h5f[path]
        if _is_plausible_dense_label_dataset(ds, length0=length0, min_ndim=min_ndim):
            return path
    return None


def _is_plausible_dense_label_dataset(
    ds: h5py.Dataset,
    *,
    length0: int,
    min_ndim: int = 3,
    min_spatial_extent: int = 8,
) -> bool:
    if not isinstance(ds, h5py.Dataset):
        return False
    if ds.ndim < min_ndim or int(ds.shape[0]) != int(length0):
        return False
    tail = [int(v) for v in ds.shape[1:] if int(v) > 1]
    if len(tail) < 2:
        return False
    spatial_dims = sorted(tail)[-2:]
    return spatial_dims[0] >= int(min_spatial_extent) and spatial_dims[1] >= int(min_spatial_extent)


def _find_recursive_dataset_with_length(
    h5f: h5py.File,
    *,
    group_prefix: str,
    length0: int,
    min_ndim: int = 3,
) -> str | None:
    if group_prefix not in h5f:
        return None
    root = h5f[group_prefix]
    if not isinstance(root, h5py.Group):
        return None
    stack: list[tuple[str, h5py.Group]] = [(group_prefix, root)]
    while stack:
        prefix, group = stack.pop()
        for key in group.keys():
            full = f"{prefix}/{key}"
            obj = group[key]
            if isinstance(obj, h5py.Group):
                stack.append((full, obj))
            elif _is_plausible_dense_label_dataset(obj, length0=length0, min_ndim=min_ndim):
                return full
    return None


def _infer_target(h5f: h5py.File, requested: str) -> TaskTarget:
    if requested in {"semantic", "depth"}:
        return requested  # type: ignore[return-value]

    sync_target = _load_h5_attr_str(h5f, "sync_target", default="").lower()
    if sync_target in {"semantic", "depth"}:
        return sync_target  # type: ignore[return-value]

    embedded = _load_h5_attr_str(h5f, "embedded_label_dataset", default="")
    if embedded == "embedded_semantics":
        return "semantic"
    if embedded == "embedded_depth":
        return "depth"

    raise ValueError("could not infer target; pass --target semantic or --target depth")


def _load_window_index(h5f: h5py.File, n_samples: int) -> np.ndarray:
    if "window_index" in h5f:
        arr = np.asarray(h5f["window_index"][()], dtype=np.int64).reshape(-1)
    else:
        arr = np.arange(n_samples, dtype=np.int64)
    if arr.shape[0] != n_samples:
        raise ValueError(f"window_index length mismatch: expected {n_samples}, got {arr.shape[0]}")
    return arr


def _resolve_label_indices(
    *,
    h5f: h5py.File,
    source_info: LabelSourceInfo,
    window_index: np.ndarray,
) -> tuple[np.ndarray, str]:
    label_index = np.asarray(window_index, dtype=np.int64)
    mode = "as_is"
    if source_info.source_kind != "embedded" or source_info.label_length <= 0 or label_index.size == 0:
        return label_index, mode

    split_start = _load_h5_attr_int(h5f, "split_source_start_index")
    split_end = _load_h5_attr_int(h5f, "split_source_end_index_exclusive")
    if split_start is None or split_end is None or int(split_end) <= int(split_start):
        return label_index, mode

    if not np.all((label_index >= int(split_start)) & (label_index < int(split_end))):
        return label_index, mode

    rebased = label_index - int(split_start)
    if not np.all((rebased >= 0) & (rebased < int(source_info.label_length))):
        return label_index, mode
    return rebased.astype(np.int64, copy=False), "rebased_split_embedded"


def _resolve_label_source(
    h5f: h5py.File,
    *,
    target: TaskTarget,
) -> LabelSourceInfo:
    if target == "semantic":
        embedded_name = "embedded_semantics"
        resolved_key = _load_h5_attr_str(h5f, "resolved_semantics_ts_source", default="")
        num_timestamps = int(_safe_attr(h5f.attrs, "num_semantic_timestamps", 0) or 0)
    else:
        embedded_name = "embedded_depth"
        resolved_key = _load_h5_attr_str(h5f, "resolved_depth_ts_source", default="")
        num_timestamps = int(_safe_attr(h5f.attrs, "num_depth_timestamps", 0) or 0)

    embedded_attr = _load_h5_attr_str(h5f, "embedded_label_dataset", default="")
    if embedded_attr == embedded_name and embedded_attr in h5f:
        ds = h5f[embedded_attr]
        if _is_plausible_dense_label_dataset(ds, length0=int(ds.shape[0]), min_ndim=3):
            return LabelSourceInfo(
                target=target,
                source_kind="embedded",
                label_dataset_path=embedded_attr,
                label_length=int(ds.shape[0]),
                embedded_dataset_name=embedded_attr,
                resolved_ts_source=resolved_key,
                num_timestamps=num_timestamps,
                missing_reason="",
            )
        embedded_shape = tuple(int(v) for v in ds.shape)
        embedded_missing_reason = (
            f"embedded label dataset {embedded_attr} has implausible shape {embedded_shape}"
        )
    else:
        embedded_missing_reason = ""

    return LabelSourceInfo(
        target=target,
        source_kind="missing",
        label_dataset_path="",
        label_length=0,
        embedded_dataset_name=embedded_attr,
        resolved_ts_source=resolved_key,
        num_timestamps=num_timestamps,
        missing_reason=(
            embedded_missing_reason
            if embedded_missing_reason != ""
            else "valid embedded label dataset is missing"
        ),
    )


def _load_label_for_window(
    *,
    pre_h5: h5py.File,
    source_info: LabelSourceInfo,
    center_window: int,
    window_index: np.ndarray,
    label_index: np.ndarray,
    label_index_mode: str,
    ignore_index: int,
    depth_scale: float,
    depth_valid_min: float,
    depth_valid_max: float,
) -> tuple[np.ndarray | None, WindowLabelInfo]:
    source_window_idx = int(window_index[center_window])
    label_idx = int(label_index[center_window])
    valid_window = (
        source_info.label_length > 0
        and 0 <= label_idx < int(source_info.label_length)
        and len(source_info.label_dataset_path) > 0
    )

    missing_reason = source_info.missing_reason
    if source_info.label_length <= 0 and missing_reason == "":
        missing_reason = "label_length is zero"
    elif not (0 <= label_idx < max(1, int(source_info.label_length))) and missing_reason == "":
        missing_reason = (
            f"label_index={label_idx} out of label range "
            f"(source_window_index={source_window_idx})"
        )

    label = None
    label_shape: tuple[int, ...] = ()
    if valid_window:
        label = np.asarray(pre_h5[source_info.label_dataset_path][label_idx])
        if label is not None:
            label_shape = tuple(int(v) for v in label.shape)

    unique_classes: list[int] = []
    depth_min = None
    depth_max = None
    depth_p05 = None
    depth_p95 = None
    depth_valid_ratio = None

    if label is not None:
        label = _squeeze_label_to_hw(label, target=source_info.target)
        if source_info.target == "semantic":
            values = np.unique(np.asarray(label, dtype=np.int64))
            unique_classes = [int(v) for v in values.tolist() if int(v) != int(ignore_index)]
        else:
            label = np.asarray(label, dtype=np.float32) * float(depth_scale)
            valid = np.isfinite(label) & (label > float(depth_valid_min)) & (label < float(depth_valid_max))
            depth_valid_ratio = float(np.mean(valid)) if valid.size > 0 else 0.0
            if np.any(valid):
                depth_values = label[valid]
                depth_min = float(depth_values.min())
                depth_max = float(depth_values.max())
                depth_p05 = float(np.percentile(depth_values, 5.0))
                depth_p95 = float(np.percentile(depth_values, 95.0))

    info = WindowLabelInfo(
        target=source_info.target,
        window_index=center_window,
        source_window_index=source_window_idx,
        label_index=label_idx,
        label_index_mode=label_index_mode,
        label_shape=label_shape,
        valid_window=bool(valid_window),
        source_kind=source_info.source_kind,
        label_dataset_path=source_info.label_dataset_path,
        missing_reason=missing_reason,
        unique_classes=unique_classes,
        depth_min=depth_min,
        depth_max=depth_max,
        depth_p05=depth_p05,
        depth_p95=depth_p95,
        depth_valid_ratio=depth_valid_ratio,
    )
    return label, info


def _resize_image(image: Image.Image, *, panel_width: int, is_label: bool) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {(w, h)}")
    scale = float(panel_width) / float(w)
    panel_height = max(1, int(round(float(h) * scale)))
    resample = _RESAMPLING.NEAREST if is_label else _RESAMPLING.BILINEAR
    return image.resize((int(panel_width), int(panel_height)), resample=resample)


def _make_placeholder(size: tuple[int, int], message: str) -> Image.Image:
    image = Image.new("RGB", size, color=(34, 34, 34))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    lines = textwrap.wrap(message, width=28) or [message]
    bbox = draw.multiline_textbbox((0, 0), "\n".join(lines), font=font, spacing=4)
    text_w = int(bbox[2] - bbox[0])
    text_h = int(bbox[3] - bbox[1])
    x = max(8, (size[0] - text_w) // 2)
    y = max(8, (size[1] - text_h) // 2)
    draw.multiline_text((x, y), "\n".join(lines), fill=(230, 230, 230), font=font, spacing=4)
    return image


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, limit // 2 - 2)
    tail = max(1, limit - head - 3)
    return f"{text[:head]}...{text[-tail:]}"


def _format_unique_classes(values: list[int], limit: int = 10) -> str:
    if len(values) == 0:
        return "none"
    if len(values) <= limit:
        return ",".join(str(v) for v in values)
    head = ",".join(str(v) for v in values[:limit])
    return f"{head},...(+{len(values) - limit})"


def _render_preview(
    *,
    file_label: str,
    target: TaskTarget,
    sync_target: str,
    voxel: np.ndarray,
    event_count: int | None,
    anchor_timestamp_us: int | None,
    source_info: LabelSourceInfo,
    window_info: WindowLabelInfo,
    label: np.ndarray | None,
    panel_width: int,
    overlay_alpha: float,
    ignore_index: int,
    split_polarity: bool,
    polarity_order: str,
    depth_valid_min: float,
    depth_valid_max: float,
) -> Image.Image:
    activity_rgb = _voxel_to_activity_rgb(voxel)
    polarity_rgb = _voxel_to_polarity_rgb(
        voxel=voxel,
        split_polarity=split_polarity,
        polarity_order=polarity_order,
    )
    target_h = int(activity_rgb.shape[0])
    target_w = int(activity_rgb.shape[1])

    label_panel: Image.Image
    overlay_rgb = activity_rgb.copy()
    extra_line = ""
    if label is None:
        label_panel = _make_placeholder(
            (int(activity_rgb.shape[1]), int(activity_rgb.shape[0])),
            message=window_info.missing_reason or "label unavailable",
        )
        extra_line = f"missing_reason={_truncate_middle(window_info.missing_reason or 'none', 72)}"
    elif target == "semantic":
        label_resized = _resize_semantic_label(np.asarray(label, dtype=np.int64), target_h=target_h, target_w=target_w)
        label_rgb = _semantic_to_rgb(label_resized, ignore_index=ignore_index)
        valid_mask = np.asarray(label_resized, dtype=np.int64) != int(ignore_index)
        overlay_rgb = _alpha_blend(activity_rgb, label_rgb, valid_mask=valid_mask, alpha=overlay_alpha)
        label_panel = Image.fromarray(label_rgb, mode="RGB")
        extra_line = (
            f"label_shape={window_info.label_shape or 'n/a'} | "
            f"classes={_format_unique_classes(window_info.unique_classes)}"
        )
    else:
        depth = np.asarray(label, dtype=np.float32)
        depth = _resize_depth_map(depth, target_h=target_h, target_w=target_w)
        valid_mask = np.isfinite(depth) & (depth > float(depth_valid_min)) & (depth < float(depth_valid_max))
        valid_mask = _resize_valid_mask(valid_mask, target_h=target_h, target_w=target_w)
        label_rgb, depth_stats = _colorize_depth(depth, valid_mask)
        overlay_rgb = _alpha_blend(activity_rgb, label_rgb, valid_mask=valid_mask, alpha=overlay_alpha)
        label_panel = Image.fromarray(label_rgb, mode="RGB")
        extra_line = (
            f"label_shape={window_info.label_shape or 'n/a'} | "
            f"depth_valid_ratio={0.0 if depth_stats['valid_ratio'] is None else depth_stats['valid_ratio']:.3f} | "
            f"depth_p05={depth_stats['p05'] if depth_stats['p05'] is not None else 'n/a'} | "
            f"depth_p95={depth_stats['p95'] if depth_stats['p95'] is not None else 'n/a'}"
        )

    panels = [
        ("activity", Image.fromarray(activity_rgb, mode="RGB"), False),
        ("polarity", Image.fromarray(polarity_rgb, mode="RGB"), False),
        ("label", label_panel, label is not None),
        ("overlay", Image.fromarray(overlay_rgb, mode="RGB"), False),
    ]

    resized_panels: list[tuple[str, Image.Image]] = []
    panel_height = 0
    for title, image, is_label in panels:
        resized = _resize_image(image, panel_width=panel_width, is_label=is_label)
        resized_panels.append((title, resized))
        panel_height = max(panel_height, resized.height)

    font = ImageFont.load_default()
    margin = 12
    panel_gap = 12
    label_gap = 18
    header_lines = [
        f"{file_label} | sync_target={sync_target or 'n/a'} | target={target} | "
        f"window={window_info.window_index} | source_idx={window_info.source_window_index} | "
        f"label_idx={window_info.label_index} | "
        f"events={event_count if event_count is not None else 'n/a'} | "
        f"anchor_us={anchor_timestamp_us if anchor_timestamp_us is not None else 'n/a'}",
        f"label_source={window_info.source_kind} | "
        f"label_ds={_truncate_middle(window_info.label_dataset_path or 'n/a', 72)} | "
        f"resolved_ts={source_info.resolved_ts_source or 'n/a'} | label_len={source_info.label_length} | "
        f"label_index_mode={window_info.label_index_mode}",
        extra_line,
    ]
    header_text = "\n".join(header_lines)
    header_bbox = ImageDraw.Draw(Image.new("RGB", (8, 8))).multiline_textbbox(
        (0, 0),
        header_text,
        font=font,
        spacing=4,
    )
    header_height = int(header_bbox[3] - header_bbox[1]) + margin * 2

    canvas_w = margin * 2 + panel_width * 2 + panel_gap
    canvas_h = header_height + margin + panel_height * 2 + label_gap * 2 + panel_gap + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((margin, margin), header_text, fill=(12, 12, 12), font=font, spacing=4)

    for i, (title, image) in enumerate(resized_panels):
        row = i // 2
        col = i % 2
        x = margin + col * (panel_width + panel_gap)
        y = header_height + margin + row * (panel_height + panel_gap + label_gap)
        canvas.paste(image, (x, y))
        draw.rectangle(
            [x - 1, y - 1, x + image.width, y + image.height],
            outline=(170, 170, 170),
            width=1,
        )
        draw.text((x, y + image.height + 4), title, fill=(12, 12, 12), font=font)
    return canvas


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "file",
        "target_requested",
        "target_used",
        "sync_target",
        "samples",
        "shape",
        "window_mode",
        "embedded_label_dataset",
        "embedded_label_source_path",
        "resolved_ts_source",
        "num_timestamps",
        "label_dataset_path",
        "label_length",
        "window_index_min",
        "window_index_max",
        "label_index_min",
        "label_index_max",
        "label_index_mode",
        "valid_windows",
        "valid_ratio",
        "label_missing_reason",
        "selection_mode_requested",
        "selection_mode_used",
        "preview_images",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in columns}
            out["shape"] = "x".join(str(v) for v in row.get("shape", []))
            out["preview_images"] = ";".join(row.get("preview_images", []))
            writer.writerow(out)


def _analyze_file(
    *,
    file_path: Path,
    dataset_root: Path | None,
    output_dir: Path,
    requested_target: str,
    num_samples: int,
    selection_mode: str,
    explicit_indices: list[int] | None,
    panel_width: int,
    overlay_alpha: float,
    ignore_index: int,
    polarity_order: str,
    depth_scale: float,
    depth_valid_min: float,
    depth_valid_max: float,
) -> dict[str, Any]:
    relative_name = _relative_or_name(file_path, dataset_root)
    stem = relative_name.replace("/", "__").replace("\\", "__")

    with h5py.File(str(file_path), "r") as pre_h5:
        voxels = pre_h5["voxels"]
        n_samples = int(voxels.shape[0])
        channels = int(voxels.shape[1])
        height = int(voxels.shape[2])
        width = int(voxels.shape[3])
        target = _infer_target(pre_h5, requested_target)
        sync_target = _load_h5_attr_str(pre_h5, "sync_target", default="")
        window_index = _load_window_index(pre_h5, n_samples)
        source_info = _resolve_label_source(pre_h5, target=target)
        label_index, label_index_mode = _resolve_label_indices(
            h5f=pre_h5,
            source_info=source_info,
            window_index=window_index,
        )
        valid_mask = np.fromiter(
            (
                1
                if (
                    source_info.label_length > 0
                    and len(source_info.label_dataset_path) > 0
                    and 0 <= int(label_idx) < int(source_info.label_length)
                )
                else 0
                for label_idx in label_index
            ),
            dtype=np.int64,
            count=n_samples,
        )
        selected_indices, selection_mode_used = _select_window_indices(
            n_samples=n_samples,
            valid_mask=valid_mask,
            num_samples=int(num_samples),
            selection_mode=selection_mode,
            explicit_indices=explicit_indices,
        )

        split_polarity = _infer_split_polarity(pre_h5, channels)
        event_counts = (
            np.asarray(pre_h5["window_event_count"][()], dtype=np.int64).reshape(-1)
            if "window_event_count" in pre_h5
            else None
        )
        anchor_timestamps = (
            np.asarray(pre_h5["anchor_timestamp_us"][()], dtype=np.int64).reshape(-1)
            if "anchor_timestamp_us" in pre_h5
            else None
        )

        preview_dir = output_dir / "previews" / stem
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_paths: list[str] = []
        preview_rows: list[dict[str, Any]] = []

        for idx in selected_indices:
            voxel = np.asarray(voxels[idx], dtype=np.float32)
            label, window_info = _load_label_for_window(
                pre_h5=pre_h5,
                source_info=source_info,
                center_window=int(idx),
                window_index=window_index,
                label_index=label_index,
                label_index_mode=label_index_mode,
                ignore_index=int(ignore_index),
                depth_scale=float(depth_scale),
                depth_valid_min=float(depth_valid_min),
                depth_valid_max=float(depth_valid_max),
            )
            preview = _render_preview(
                file_label=relative_name,
                target=target,
                sync_target=sync_target,
                voxel=voxel,
                event_count=(None if event_counts is None or idx >= len(event_counts) else int(event_counts[idx])),
                anchor_timestamp_us=(
                    None if anchor_timestamps is None or idx >= len(anchor_timestamps) else int(anchor_timestamps[idx])
                ),
                source_info=source_info,
                window_info=window_info,
                label=label,
                panel_width=int(panel_width),
                overlay_alpha=float(overlay_alpha),
                ignore_index=int(ignore_index),
                split_polarity=split_polarity,
                polarity_order=polarity_order,
                depth_valid_min=float(depth_valid_min),
                depth_valid_max=float(depth_valid_max),
            )
            out_path = preview_dir / f"window_{int(idx):06d}.png"
            preview.save(out_path)
            preview_paths.append(str(out_path))
            preview_rows.append(
                {
                    "window_index": int(idx),
                    "source_window_index": int(window_info.source_window_index),
                    "label_index": int(window_info.label_index),
                    "label_index_mode": window_info.label_index_mode,
                    "label_shape": list(window_info.label_shape),
                    "valid_window": int(window_info.valid_window),
                    "source_kind": window_info.source_kind,
                    "label_dataset_path": window_info.label_dataset_path,
                    "missing_reason": window_info.missing_reason,
                    "unique_classes": window_info.unique_classes,
                    "depth_min": window_info.depth_min,
                    "depth_max": window_info.depth_max,
                    "depth_p05": window_info.depth_p05,
                    "depth_p95": window_info.depth_p95,
                    "depth_valid_ratio": window_info.depth_valid_ratio,
                    "preview_path": str(out_path),
                }
            )

        result = {
            "file": str(file_path),
            "relative_file": relative_name,
            "target_requested": requested_target,
            "target_used": target,
            "sync_target": sync_target,
            "samples": n_samples,
            "shape": [n_samples, channels, height, width],
            "window_mode": _load_h5_attr_str(pre_h5, "window_mode", default=""),
            "embedded_label_dataset": _load_h5_attr_str(pre_h5, "embedded_label_dataset", default=""),
            "embedded_label_source_path": _load_h5_attr_str(pre_h5, "embedded_label_source_path", default=""),
            "resolved_ts_source": source_info.resolved_ts_source,
            "num_timestamps": int(source_info.num_timestamps),
            "label_dataset_path": source_info.label_dataset_path,
            "label_length": int(source_info.label_length),
            "window_index_min": (int(window_index.min()) if window_index.size > 0 else None),
            "window_index_max": (int(window_index.max()) if window_index.size > 0 else None),
            "label_index_min": (int(label_index.min()) if label_index.size > 0 else None),
            "label_index_max": (int(label_index.max()) if label_index.size > 0 else None),
            "label_index_mode": label_index_mode,
            "valid_windows": int(np.count_nonzero(valid_mask > 0)),
            "valid_ratio": float(np.mean(valid_mask > 0)) if valid_mask.size > 0 else 0.0,
            "selection_mode_requested": selection_mode,
            "selection_mode_used": selection_mode_used,
            "preview_images": preview_paths,
            "preview_rows": preview_rows,
            "label_missing_reason": source_info.missing_reason,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        "Visualize M3ED semantic/depth sync debug panels from preprocessed voxel H5 files."
    )
    parser.add_argument("--input_path", type=Path, default=None, help="Single preprocessed voxel H5.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory containing preprocessed voxel H5.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan --dataset_root.",
    )
    parser.add_argument("--file_pattern", type=str, default="*.h5", help="Glob under --dataset_root.")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on files to analyze.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tmp/m3ed_sync_debug"),
        help="Where preview PNGs and reports are written.",
    )
    parser.add_argument(
        "--target",
        choices=["auto", "semantic", "depth"],
        default="auto",
        help="Which label target to inspect. auto uses sync_target / embedded label metadata.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="How many windows to preview per file when --window_indices is not set.",
    )
    parser.add_argument(
        "--window_indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit window indices to preview.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["valid", "invalid", "all"],
        default="valid",
        help="Which windows to sample when --window_indices is not set.",
    )
    parser.add_argument("--panel_width", type=int, default=320, help="Width of each panel in the output grid.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Overlay alpha for label-on-activity.")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore label value for semantic.")
    parser.add_argument(
        "--polarity_order",
        choices=["negpos", "posneg"],
        default="negpos",
        help="How split-polarity channels are ordered inside the voxel tensor.",
    )
    parser.add_argument("--depth_scale", type=float, default=1.0, help="Scale factor applied before depth visualization.")
    parser.add_argument("--depth_valid_min", type=float, default=0.0, help="Minimum valid depth.")
    parser.add_argument("--depth_valid_max", type=float, default=1e9, help="Maximum valid depth.")
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional JSON report path. Default: <output_dir>/summary.json",
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=None,
        help="Optional CSV report path. Default: <output_dir>/summary.csv",
    )
    args = parser.parse_args()

    input_path = None if args.input_path is None else Path(args.input_path).expanduser().resolve()
    dataset_root = None if args.dataset_root is None else Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _select_files(
        input_path=input_path,
        dataset_root=dataset_root,
        recursive=bool(args.recursive),
        file_pattern=str(args.file_pattern),
        max_files=args.max_files,
    )
    if len(files) == 0:
        raise FileNotFoundError("no M3ED voxel h5 files found")

    results: list[dict[str, Any]] = []
    for file_path in files:
        result = _analyze_file(
            file_path=file_path,
            dataset_root=dataset_root,
            output_dir=output_dir,
            requested_target=str(args.target),
            num_samples=int(args.num_samples),
            selection_mode=str(args.selection_mode),
            explicit_indices=args.window_indices,
            panel_width=int(args.panel_width),
            overlay_alpha=float(args.overlay_alpha),
            ignore_index=int(args.ignore_index),
            polarity_order=str(args.polarity_order),
            depth_scale=float(args.depth_scale),
            depth_valid_min=float(args.depth_valid_min),
            depth_valid_max=float(args.depth_valid_max),
        )
        results.append(result)
        print(
            "[M3ED DEBUG] "
            f"{result['relative_file']} | target={result['target_used']} | samples={result['samples']} | "
            f"valid={result['valid_windows']}/{result['samples']} ({result['valid_ratio']:.3f}) | "
            f"selection={result['selection_mode_used']} | previews={len(result['preview_images'])}"
        )

    report_json = (
        Path(args.report_json).expanduser().resolve()
        if args.report_json is not None
        else output_dir / "summary.json"
    )
    report_csv = (
        Path(args.report_csv).expanduser().resolve()
        if args.report_csv is not None
        else output_dir / "summary.csv"
    )

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(report_csv, results)
    print(f"[M3ED DEBUG] wrote JSON report: {report_json}")
    print(f"[M3ED DEBUG] wrote CSV report:  {report_csv}")


if __name__ == "__main__":
    main()
