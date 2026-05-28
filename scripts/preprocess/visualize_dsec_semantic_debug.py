from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont


try:
    _RESAMPLING = Image.Resampling
except AttributeError:  # Pillow < 9.1
    _RESAMPLING = Image


@dataclass(frozen=True)
class LabelDebugInfo:
    available: int
    timestamp_us: int | None
    delta_us: int | None
    relpath: str
    source: str
    missing_reason: str
    unique_classes: list[int]
    label_shape: tuple[int, int] | None


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


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            return "voxels" in h5f and h5f["voxels"].ndim == 4 and int(h5f["voxels"].shape[0]) > 0
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
    available_mask: np.ndarray,
    num_samples: int,
    selection_mode: str,
    explicit_indices: list[int] | None,
) -> tuple[list[int], str]:
    all_indices = np.arange(n_samples, dtype=np.int64)
    if explicit_indices:
        selected = sorted(set(int(i) for i in explicit_indices if 0 <= int(i) < n_samples))
        return selected, "explicit"

    if selection_mode == "available":
        eligible = all_indices[available_mask > 0]
    elif selection_mode == "missing":
        eligible = all_indices[available_mask <= 0]
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


def _label_palette() -> np.ndarray:
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


def _squeeze_label_to_hw(label: np.ndarray) -> np.ndarray:
    if label.ndim == 2:
        return label
    if label.ndim == 3 and label.shape[0] == 1:
        return label[0]
    if label.ndim == 3 and label.shape[-1] == 1:
        return label[..., 0]
    raise ValueError(f"unsupported label shape={label.shape}; expected HxW or singleton-channel variant")


def _label_to_rgb(label: np.ndarray, *, ignore_index: int) -> np.ndarray:
    palette = _label_palette()
    label = np.asarray(label, dtype=np.int64)
    rgb = palette[np.clip(label, 0, 255)]
    rgb[label == int(ignore_index)] = palette[int(ignore_index)]
    return rgb


def _alpha_blend(base_rgb: np.ndarray, label_rgb: np.ndarray, valid_mask: np.ndarray, alpha: float) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32)
    label = np.asarray(label_rgb, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if base.shape != label.shape:
        raise ValueError(f"shape mismatch for alpha blend: base={base.shape}, label={label.shape}")
    if mask.shape != base.shape[:2]:
        raise ValueError(f"mask shape mismatch for alpha blend: mask={mask.shape}, expected={base.shape[:2]}")
    out = base.copy()
    out[mask] = (1.0 - float(alpha)) * base[mask] + float(alpha) * label[mask]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _format_hw(shape: tuple[int, int] | None) -> str:
    if shape is None or len(shape) != 2:
        return "n/a"
    return f"{int(shape[0])}x{int(shape[1])}"


def _align_semantic_label_to_hw(
    label: np.ndarray,
    *,
    target_h: int,
    target_w: int,
    ignore_index: int,
) -> np.ndarray:
    arr = np.asarray(label, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"expected semantic label [H,W], got shape={arr.shape}")
    if arr.shape == (int(target_h), int(target_w)):
        return arr

    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])

    # DSEC semantic labels are commonly shorter in height than the event frame.
    # Preserve the original top-left alignment and pad/crop only on the bottom/right.
    if src_w == int(target_w) or src_h == int(target_h) or (src_h <= int(target_h) and src_w <= int(target_w)):
        out = np.full((int(target_h), int(target_w)), int(ignore_index), dtype=arr.dtype)
        copy_h = min(src_h, int(target_h))
        copy_w = min(src_w, int(target_w))
        out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
        return out

    image = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L")
    resized = image.resize((int(target_w), int(target_h)), resample=_RESAMPLING.NEAREST)
    return np.asarray(resized, dtype=np.int64)


def _load_label_for_window(h5f: h5py.File, window_idx: int, ignore_index: int) -> tuple[np.ndarray | None, LabelDebugInfo]:
    n_samples = int(h5f["voxels"].shape[0])
    avail_arr = (
        np.asarray(h5f["segmentation_available"][()], dtype=np.int64).reshape(-1)
        if "segmentation_available" in h5f
        else np.ones((n_samples,), dtype=np.int64)
    )
    ts_arr = (
        np.asarray(h5f["segmentation_timestamp_us"][()], dtype=np.int64).reshape(-1)
        if "segmentation_timestamp_us" in h5f
        else None
    )
    delta_arr = (
        np.asarray(h5f["segmentation_time_delta_us"][()], dtype=np.int64).reshape(-1)
        if "segmentation_time_delta_us" in h5f
        else None
    )
    relpaths = ()
    if "segmentation_relpath" in h5f:
        seg_rel = h5f["segmentation_relpath"][()]
        relpaths = tuple(_decode_h5_string(v).strip() for v in seg_rel.tolist())

    available = int(avail_arr[window_idx]) if window_idx < len(avail_arr) else 0
    timestamp_us = int(ts_arr[window_idx]) if ts_arr is not None and window_idx < len(ts_arr) else None
    delta_us = int(delta_arr[window_idx]) if delta_arr is not None and window_idx < len(delta_arr) else None
    relpath = relpaths[window_idx] if window_idx < len(relpaths) else ""

    embedded_path = _load_h5_attr_str(h5f, "embedded_label_dataset", default="")
    label = None
    source = ""
    missing_reason = ""

    if embedded_path == "embedded_segmentation" and embedded_path in h5f:
        ds = h5f[embedded_path]
        if ds.ndim >= 3 and int(ds.shape[0]) > window_idx:
            label = _squeeze_label_to_hw(np.asarray(ds[window_idx]))
            source = "embedded_segmentation"

    if label is None and missing_reason == "":
        if embedded_path != "embedded_segmentation":
            missing_reason = "embedded_segmentation dataset is unavailable"
        else:
            missing_reason = "label unavailable"

    unique_classes: list[int] = []
    if label is not None:
        values = np.unique(np.asarray(label, dtype=np.int64))
        unique_classes = [int(v) for v in values.tolist() if int(v) != int(ignore_index)]

    info = LabelDebugInfo(
        available=available,
        timestamp_us=timestamp_us,
        delta_us=delta_us,
        relpath=relpath,
        source=source,
        missing_reason=missing_reason,
        unique_classes=unique_classes,
        label_shape=(None if label is None else (int(label.shape[0]), int(label.shape[1]))),
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


def _format_unique_classes(values: list[int], limit: int = 10) -> str:
    if len(values) == 0:
        return "none"
    if len(values) <= limit:
        return ",".join(str(v) for v in values)
    head = ",".join(str(v) for v in values[:limit])
    return f"{head},...(+{len(values) - limit})"


def _truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = max(1, limit // 2 - 2)
    tail = max(1, limit - head - 3)
    return f"{text[:head]}...{text[-tail:]}"


def _render_preview(
    *,
    file_label: str,
    window_idx: int,
    voxel: np.ndarray,
    event_count: int | None,
    anchor_timestamp_us: int | None,
    label: np.ndarray | None,
    label_info: LabelDebugInfo,
    panel_width: int,
    ignore_index: int,
    overlay_alpha: float,
    split_polarity: bool,
    polarity_order: str,
) -> Image.Image:
    activity_rgb = _voxel_to_activity_rgb(voxel)
    polarity_rgb = _voxel_to_polarity_rgb(
        voxel,
        split_polarity=split_polarity,
        polarity_order=polarity_order,
    )
    target_h = int(activity_rgb.shape[0])
    target_w = int(activity_rgb.shape[1])
    aligned_label_shape: tuple[int, int] | None = None

    if label is not None:
        label = _align_semantic_label_to_hw(
            label,
            target_h=target_h,
            target_w=target_w,
            ignore_index=ignore_index,
        )
        aligned_label_shape = (int(label.shape[0]), int(label.shape[1]))
        label_rgb = _label_to_rgb(label, ignore_index=ignore_index)
        valid_mask = np.asarray(label, dtype=np.int64) != int(ignore_index)
        overlay_rgb = _alpha_blend(activity_rgb, label_rgb, valid_mask=valid_mask, alpha=overlay_alpha)
        label_panel = Image.fromarray(label_rgb, mode="RGB")
    else:
        overlay_rgb = activity_rgb.copy()
        label_panel = _make_placeholder(
            (int(activity_rgb.shape[1]), int(activity_rgb.shape[0])),
            message=label_info.missing_reason or "label unavailable",
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
        f"{file_label} | window={window_idx} | avail={label_info.available} | "
        f"events={event_count if event_count is not None else 'n/a'} | "
        f"anchor_us={anchor_timestamp_us if anchor_timestamp_us is not None else 'n/a'} | "
        f"delta_us={label_info.delta_us if label_info.delta_us is not None else 'n/a'}",
        f"label_source={label_info.source or 'missing'} | "
        f"relpath={_truncate_middle(label_info.relpath, 72) or 'n/a'}",
        f"classes={_format_unique_classes(label_info.unique_classes)} | "
        f"label_shape={_format_hw(label_info.label_shape)}->{_format_hw(aligned_label_shape)} | "
        f"missing_reason={_truncate_middle(label_info.missing_reason or 'none', 72)}",
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
        "samples",
        "shape",
        "window_mode",
        "sync_segmentation",
        "embedded_label_dataset",
        "labeled_windows",
        "labeled_ratio",
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
    num_samples: int,
    selection_mode: str,
    explicit_indices: list[int] | None,
    panel_width: int,
    overlay_alpha: float,
    ignore_index: int,
    polarity_order: str,
) -> dict[str, Any]:
    relative_name = _relative_or_name(file_path, dataset_root)
    stem = relative_name.replace("/", "__").replace("\\", "__")

    with h5py.File(str(file_path), "r") as h5f:
        voxels = h5f["voxels"]
        n_samples = int(voxels.shape[0])
        channels = int(voxels.shape[1])
        height = int(voxels.shape[2])
        width = int(voxels.shape[3])

        available = (
            np.asarray(h5f["segmentation_available"][()], dtype=np.int64).reshape(-1)
            if "segmentation_available" in h5f
            else np.ones((n_samples,), dtype=np.int64)
        )
        if available.shape[0] != n_samples:
            raise ValueError(
                f"segmentation_available length mismatch: expected {n_samples}, got {available.shape[0]}"
            )

        selected_indices, selection_mode_used = _select_window_indices(
            n_samples=n_samples,
            available_mask=available,
            num_samples=int(num_samples),
            selection_mode=selection_mode,
            explicit_indices=explicit_indices,
        )

        split_polarity = _infer_split_polarity(h5f, channels)
        event_counts = (
            np.asarray(h5f["window_event_count"][()], dtype=np.int64).reshape(-1)
            if "window_event_count" in h5f
            else None
        )
        anchor_timestamps = (
            np.asarray(h5f["anchor_timestamp_us"][()], dtype=np.int64).reshape(-1)
            if "anchor_timestamp_us" in h5f
            else None
        )

        preview_dir = output_dir / "previews" / stem
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_paths: list[str] = []
        preview_rows: list[dict[str, Any]] = []
        for idx in selected_indices:
            voxel = np.asarray(voxels[idx], dtype=np.float32)
            label, label_info = _load_label_for_window(h5f, idx, ignore_index=ignore_index)
            preview = _render_preview(
                file_label=relative_name,
                window_idx=int(idx),
                voxel=voxel,
                event_count=(None if event_counts is None or idx >= len(event_counts) else int(event_counts[idx])),
                anchor_timestamp_us=(
                    None if anchor_timestamps is None or idx >= len(anchor_timestamps) else int(anchor_timestamps[idx])
                ),
                label=label,
                label_info=label_info,
                panel_width=int(panel_width),
                ignore_index=int(ignore_index),
                overlay_alpha=float(overlay_alpha),
                split_polarity=split_polarity,
                polarity_order=polarity_order,
            )
            out_path = preview_dir / f"window_{int(idx):06d}.png"
            preview.save(out_path)
            preview_paths.append(str(out_path))
            preview_rows.append(
                {
                    "window_index": int(idx),
                    "available": int(label_info.available),
                    "relpath": label_info.relpath,
                    "missing_reason": label_info.missing_reason,
                    "unique_classes": label_info.unique_classes,
                    "preview_path": str(out_path),
                }
            )

        result = {
            "file": str(file_path),
            "relative_file": relative_name,
            "samples": n_samples,
            "shape": [n_samples, channels, height, width],
            "window_mode": _load_h5_attr_str(h5f, "window_mode", default=""),
            "sync_segmentation": int(_safe_attr(h5f.attrs, "sync_segmentation", 0) or 0),
            "embedded_label_dataset": _load_h5_attr_str(h5f, "embedded_label_dataset", default=""),
            "labeled_windows": int(np.count_nonzero(available > 0)),
            "labeled_ratio": float(np.mean(available > 0)) if available.size > 0 else 0.0,
            "label_missing_reason": (
                next((row["missing_reason"] for row in preview_rows if row["missing_reason"]), "")
                if len(preview_rows) > 0
                else ""
            ),
            "selection_mode_requested": selection_mode,
            "selection_mode_used": selection_mode_used,
            "preview_images": preview_paths,
            "preview_rows": preview_rows,
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        "Visualize DSEC semantic debug panels from preprocessed voxel H5 files."
    )
    parser.add_argument("--input_path", type=Path, default=None, help="Single preprocessed voxel H5.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory containing voxel H5 files.")
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
        default=Path("tmp/dsec_semantic_debug"),
        help="Where preview PNGs and reports are written.",
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
        choices=["available", "missing", "all"],
        default="available",
        help="Which windows to sample when --window_indices is not set.",
    )
    parser.add_argument("--panel_width", type=int, default=320, help="Width of each panel in the output grid.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Overlay alpha for label-on-activity.")
    parser.add_argument("--ignore_index", type=int, default=255, help="Ignore label value.")
    parser.add_argument(
        "--polarity_order",
        choices=["negpos", "posneg"],
        default="negpos",
        help="How split-polarity channels are ordered inside the voxel tensor.",
    )
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
        raise FileNotFoundError("no voxel h5 files found")

    results: list[dict[str, Any]] = []
    for file_path in files:
        result = _analyze_file(
            file_path=file_path,
            dataset_root=dataset_root,
            output_dir=output_dir,
            num_samples=int(args.num_samples),
            selection_mode=str(args.selection_mode),
            explicit_indices=args.window_indices,
            panel_width=int(args.panel_width),
            overlay_alpha=float(args.overlay_alpha),
            ignore_index=int(args.ignore_index),
            polarity_order=str(args.polarity_order),
        )
        results.append(result)
        print(
            "[DSEC DEBUG] "
            f"{result['relative_file']} | samples={result['samples']} | "
            f"labeled={result['labeled_windows']}/{result['samples']} "
            f"({result['labeled_ratio']:.3f}) | previews={len(result['preview_images'])} | "
            f"selection={result['selection_mode_used']}"
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
    print(f"[DSEC DEBUG] wrote JSON report: {report_json}")
    print(f"[DSEC DEBUG] wrote CSV report:  {report_csv}")


if __name__ == "__main__":
    main()
