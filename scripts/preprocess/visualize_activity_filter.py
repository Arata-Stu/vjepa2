from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - import guard for environments without h5py
    h5py = None

try:
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - gzip-only files can still work without it
    hdf5plugin = None

import numpy as np


DEFAULT_ACTIVE_METRIC = "window_active_pixel_ratio"
DEFAULT_SCORE_METRIC = "window_activity_score"


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required to analyze H5 files. Install project dependencies first.")


def _safe_attr(attrs: Any, key: str, default: Any = None) -> Any:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_voxel_h5(path: Path) -> bool:
    _require_h5py()
    try:
        with h5py.File(str(path), "r") as h5f:
            return "voxels" in h5f and h5f["voxels"].ndim == 4
    except Exception:
        return False


def _select_files(input_path: Path | None, dataset_root: Path | None, recursive: bool) -> list[Path]:
    if input_path is not None and dataset_root is not None:
        raise ValueError("use either --input_path or --dataset_root, not both")
    if input_path is None and dataset_root is None:
        raise ValueError("either --input_path or --dataset_root is required")
    if input_path is not None:
        return [input_path]
    assert dataset_root is not None
    pattern = "**/*.h5" if recursive else "*.h5"
    return sorted([p for p in dataset_root.glob(pattern) if p.is_file()])


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _infer_dataset_family(file_path: Path, representation: str) -> str:
    rep = str(representation).strip().lower()
    if rep == "event_voxel_grid_m3ed":
        return "m3ed"
    if rep == "event_voxel_grid_1mpx":
        return "1mpx"
    if rep == "event_voxel_grid_eventscape":
        return "eventscape"
    if rep == "event_voxel_grid":
        return "dsec"

    lower_path = str(file_path).lower()
    if "m3ed" in lower_path:
        return "m3ed"
    if "1mpx" in lower_path:
        return "1mpx"
    if "eventscape" in lower_path:
        return "eventscape"
    if "dsec" in lower_path:
        return "dsec"
    return "other"


def _read_1d_dataset(dset: Any, chunk_size: int) -> np.ndarray:
    n = int(dset.shape[0])
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    parts: list[np.ndarray] = []
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        arr = np.asarray(dset[start:end], dtype=np.float32).reshape(-1)
        if arr.size > 0:
            parts.append(arr)
    if len(parts) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def _infer_split_polarity(h5f: h5py.File, channels: int) -> bool:
    attr = _safe_attr(h5f.attrs, "split_polarity", None)
    if attr is not None:
        return bool(int(attr))
    return channels % 2 == 0


def _vote_score_map(
    voxel: np.ndarray,
    split_polarity: bool,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
) -> np.ndarray:
    if voxel.ndim != 3:
        raise ValueError(f"voxel must be (C,H,W), got shape={voxel.shape}")

    channels = voxel.shape[0]
    if channels == 0:
        return np.zeros((voxel.shape[1], voxel.shape[2]), dtype=np.float32)

    if split_polarity and channels > 1 and channels % 2 == 0:
        half = channels // 2
        if polarity_order == "negpos":
            neg_bins = np.asarray(voxel[:half], dtype=np.float32)
            pos_bins = np.asarray(voxel[half:], dtype=np.float32)
        elif polarity_order == "posneg":
            pos_bins = np.asarray(voxel[:half], dtype=np.float32)
            neg_bins = np.asarray(voxel[half:], dtype=np.float32)
        else:
            raise ValueError(f"unsupported polarity_order: {polarity_order}")

        if vote_use_abs_for_split_polarity:
            pos_bins = np.abs(pos_bins)
            neg_bins = np.abs(neg_bins)

        vote_bins = np.sign(pos_bins - neg_bins)
        vote_score = vote_bins.sum(axis=0)
    else:
        vote_score = np.sign(np.asarray(voxel, dtype=np.float32)).sum(axis=0)
    return vote_score


def _voxel_to_rgb(
    voxel: np.ndarray,
    split_polarity: bool,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
    tie_epsilon: float,
) -> np.ndarray:
    vote_score = _vote_score_map(
        voxel=voxel,
        split_polarity=split_polarity,
        polarity_order=polarity_order,
        vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
    )
    height, width = vote_score.shape
    img = np.full((height, width, 3), 127, dtype=np.uint8)
    eps = float(max(0.0, tie_epsilon))
    img[vote_score > eps] = 255
    img[vote_score < -eps] = 0
    return img


def _fit_indices_length(indices: np.ndarray, target_len: int, last_valid: int) -> np.ndarray:
    if indices.size >= target_len:
        pos = np.linspace(0, indices.size - 1, num=target_len, dtype=np.float64)
        pos = np.round(pos).astype(np.int64)
        return indices[pos]
    pad = np.full((target_len - indices.size,), int(last_valid), dtype=np.int64)
    return np.concatenate([indices, pad], axis=0)


def _enumerate_clip_starts(total_windows: int, frames_per_clip: int, frame_step: int, clip_stride: int) -> np.ndarray:
    if total_windows <= 0:
        return np.empty((0,), dtype=np.int64)
    clip_span = max(1, int(frames_per_clip) * int(frame_step))
    if total_windows <= clip_span:
        return np.array([0], dtype=np.int64)
    max_start = total_windows - clip_span
    starts = np.arange(0, max_start + 1, max(1, int(clip_stride)), dtype=np.int64)
    if starts.size == 0 or int(starts[-1]) != int(max_start):
        starts = np.concatenate([starts, np.array([max_start], dtype=np.int64)], axis=0)
    return np.unique(starts)


def _clip_indices_from_start(start: int, total_windows: int, frames_per_clip: int, frame_step: int) -> np.ndarray:
    if total_windows <= 0:
        return np.zeros((frames_per_clip,), dtype=np.int64)
    clip_span = max(1, int(frames_per_clip) * int(frame_step))
    if total_windows > clip_span:
        return (int(start) + np.arange(0, clip_span, int(frame_step), dtype=np.int64)).astype(np.int64, copy=False)

    local = np.arange(0, total_windows, int(frame_step), dtype=np.int64)
    if local.size == 0:
        local = np.array([0], dtype=np.int64)
    local = _fit_indices_length(local, target_len=int(frames_per_clip), last_valid=max(0, total_windows - 1))
    np.clip(local, 0, total_windows - 1, out=local)
    return local.astype(np.int64, copy=False)


def _read_voxels_by_indices(voxels_ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("indices cannot be empty")
    if indices.size > 1 and np.all(indices[1:] - indices[:-1] == 1):
        start = int(indices[0])
        end = int(indices[-1]) + 1
        arr = np.asarray(voxels_ds[start:end], dtype=np.float32)
    else:
        arr = np.stack([np.asarray(voxels_ds[int(i)], dtype=np.float32) for i in indices], axis=0)
    if arr.ndim != 4:
        raise ValueError(f"unexpected voxel clip shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _normalized_margin(value: float, threshold: float) -> float:
    if threshold <= 0:
        return float(value) - float(threshold)
    return (float(value) - float(threshold)) / float(threshold)


def _update_sorted_examples(
    bucket: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    k: int,
    key_name: str,
    reverse: bool,
) -> None:
    if k <= 0:
        return
    bucket.append(dict(row))
    bucket.sort(key=lambda x: float(x[key_name]), reverse=reverse)
    if len(bucket) > k:
        del bucket[k:]


def _reservoir_update(
    reservoir: list[dict[str, Any]],
    row: dict[str, Any],
    *,
    seen_count: int,
    max_items: int,
    rng: random.Random,
) -> None:
    if max_items <= 0:
        return
    if len(reservoir) < max_items:
        reservoir.append(dict(row))
        return
    slot = rng.randint(0, seen_count - 1)
    if slot < max_items:
        reservoir[slot] = dict(row)


def _build_reason_string(*, mean_active_ok: bool, active_frac_ok: bool, mean_score_ok: bool, use_score: bool) -> str:
    reasons: list[str] = []
    if not mean_active_ok:
        reasons.append("clip_mean_active")
    if not active_frac_ok:
        reasons.append("clip_active_frac")
    if use_score and not mean_score_ok:
        reasons.append("clip_mean_score")
    if len(reasons) == 0:
        return "keep"
    return "|".join(reasons)


def _compute_clip_metrics_vectorized(
    active_values: np.ndarray,
    score_values: np.ndarray | None,
    *,
    frames_per_clip: int,
    frame_step: int,
    clip_stride: int,
    active_eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    starts = _enumerate_clip_starts(
        total_windows=int(active_values.size),
        frames_per_clip=int(frames_per_clip),
        frame_step=int(frame_step),
        clip_stride=int(clip_stride),
    )
    if starts.size == 0:
        return starts, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), None

    if int(frame_step) == 1 and int(active_values.size) >= int(frames_per_clip):
        span = int(frames_per_clip)
        csum_active = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(active_values, dtype=np.float64)])
        full_mean_active = (csum_active[span:] - csum_active[:-span]) / float(span)
        active_binary = (active_values > float(active_eps)).astype(np.float32)
        csum_binary = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(active_binary, dtype=np.float64)])
        full_frac_active = (csum_binary[span:] - csum_binary[:-span]) / float(span)
        mean_active = full_mean_active[starts].astype(np.float32, copy=False)
        frac_active = full_frac_active[starts].astype(np.float32, copy=False)
        mean_score = None
        if score_values is not None:
            csum_score = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(score_values, dtype=np.float64)])
            full_mean_score = (csum_score[span:] - csum_score[:-span]) / float(span)
            mean_score = full_mean_score[starts].astype(np.float32, copy=False)
        return starts, mean_active, frac_active, mean_score

    mean_active = np.empty((starts.size,), dtype=np.float32)
    frac_active = np.empty((starts.size,), dtype=np.float32)
    mean_score = None if score_values is None else np.empty((starts.size,), dtype=np.float32)
    for idx, start in enumerate(starts.tolist()):
        indices = _clip_indices_from_start(
            start=int(start),
            total_windows=int(active_values.size),
            frames_per_clip=int(frames_per_clip),
            frame_step=int(frame_step),
        )
        clip_active = active_values[indices]
        mean_active[idx] = float(clip_active.mean()) if clip_active.size > 0 else 0.0
        frac_active[idx] = float(np.mean(clip_active > float(active_eps))) if clip_active.size > 0 else 0.0
        if mean_score is not None and score_values is not None:
            clip_score = score_values[indices]
            mean_score[idx] = float(clip_score.mean()) if clip_score.size > 0 else 0.0
    return starts, mean_active, frac_active, mean_score


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_float(value: float | None, fmt: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(float(value), fmt)


def _make_hist_counts(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    counts, _ = np.histogram(values, bins=int(bins), range=value_range)
    return counts.astype(np.int64, copy=False)


def _build_keep_drop_histogram_svg(
    *,
    output_path: Path,
    metrics: dict[str, dict[str, Any]],
    value_range: tuple[float, float],
    chart_title: str,
) -> None:
    names = list(metrics.keys())
    if len(names) == 0:
        return

    width = 1200
    panel_height = 300
    top_margin = 48
    left_margin = 72
    right_margin = 28
    bottom_margin = 58
    inner_gap = 30
    height = top_margin + len(names) * panel_height + (len(names) - 1) * inner_gap + 24
    plot_width = width - left_margin - right_margin
    plot_height = panel_height - bottom_margin - 52
    min_x, max_x = value_range

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left_margin}" y="28" font-size="24" font-family="monospace" fill="#111111">{_svg_escape(chart_title)}</text>',
    ]

    for metric_idx, metric_name in enumerate(names):
        panel_top = top_margin + metric_idx * (panel_height + inner_gap)
        plot_left = left_margin
        plot_top = panel_top + 34
        plot_bottom = plot_top + plot_height
        plot_right = plot_left + plot_width
        payload = metrics[metric_name]
        edges = payload["edges"]
        all_frac = payload["all_frac"]
        keep_frac = payload["keep_frac"]
        drop_frac = payload["drop_frac"]
        keep_ratio = payload["keep_ratio"]
        drop_ratio = payload["drop_ratio"]
        y_max = max(float(all_frac.max()) if all_frac.size > 0 else 0.0, 1e-6)

        lines.append(
            f'<text x="{plot_left}" y="{panel_top + 18}" font-size="18" font-family="monospace" fill="#111111">{_svg_escape(metric_name)}</text>'
        )
        subtitle = f"keep_ratio={keep_ratio:.4f}  drop_ratio={drop_ratio:.4f}"
        lines.append(
            f'<text x="{plot_left}" y="{panel_top + 38}" font-size="13" font-family="monospace" fill="#444444">{_svg_escape(subtitle)}</text>'
        )
        lines.append(
            f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#dddddd" stroke-width="1"/>'
        )

        for tick_idx in range(6):
            frac = tick_idx / 5.0
            y = plot_bottom - frac * plot_height
            label = f"{frac * y_max:.3f}"
            lines.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="#eeeeee" stroke-width="1"/>')
            lines.append(
                f'<text x="{plot_left - 8}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="monospace" fill="#666666">{_svg_escape(label)}</text>'
            )

        for tick_idx in range(11):
            frac = tick_idx / 10.0
            x = plot_left + frac * plot_width
            label = min_x + frac * (max_x - min_x)
            lines.append(f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom + 6}" stroke="#888888" stroke-width="1"/>')
            lines.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 22}" font-size="11" text-anchor="middle" font-family="monospace" fill="#666666">{_svg_escape(f"{label:.2f}")}</text>'
            )

        for bin_idx in range(len(all_frac)):
            x0 = plot_left + (edges[bin_idx] - min_x) / max(max_x - min_x, 1e-12) * plot_width
            x1 = plot_left + (edges[bin_idx + 1] - min_x) / max(max_x - min_x, 1e-12) * plot_width
            bar_width = max(x1 - x0 - 1.0, 0.2)
            all_h = (float(all_frac[bin_idx]) / y_max) * plot_height
            drop_h = (float(drop_frac[bin_idx]) / y_max) * plot_height
            keep_h = (float(keep_frac[bin_idx]) / y_max) * plot_height
            lines.append(
                f'<rect x="{x0:.2f}" y="{plot_bottom - all_h:.2f}" width="{bar_width:.2f}" height="{all_h:.2f}" fill="#d0d5dd" opacity="0.8"/>'
            )
            lines.append(
                f'<rect x="{x0:.2f}" y="{plot_bottom - drop_h:.2f}" width="{bar_width:.2f}" height="{drop_h:.2f}" fill="#d92d20" opacity="0.75"/>'
            )
            lines.append(
                f'<rect x="{x0:.2f}" y="{plot_bottom - keep_h:.2f}" width="{bar_width:.2f}" height="{keep_h:.2f}" fill="#16a34a" opacity="0.75"/>'
            )

        legend_x = plot_right - 240
        legend_y = plot_top + 16
        legend = [
            ("all", "#d0d5dd"),
            ("drop", "#d92d20"),
            ("keep", "#16a34a"),
        ]
        for idx, (label, color) in enumerate(legend):
            y = legend_y + idx * 18
            lines.append(f'<rect x="{legend_x}" y="{y - 10}" width="18" height="12" fill="{color}" opacity="0.8"/>')
            lines.append(
                f'<text x="{legend_x + 24}" y="{y}" font-size="11" font-family="monospace" fill="#555555">{_svg_escape(label)}</text>'
            )

        lines.append(
            f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{plot_bottom + 42}" font-size="12" text-anchor="middle" font-family="monospace" fill="#666666">clip metric value</text>'
        )
        lines.append(
            f'<text transform="translate({plot_left - 54},{(plot_top + plot_bottom) / 2:.2f}) rotate(-90)" font-size="12" text-anchor="middle" font-family="monospace" fill="#666666">fraction of clips</text>'
        )

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _render_clip_strip(
    *,
    frames: list[np.ndarray],
    title: str,
    subtitle_lines: list[str],
    frame_display_height: int,
) -> Any:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required for clip-strip rendering. Install via `pip install pillow`.") from exc

    if len(frames) == 0:
        raise ValueError("frames cannot be empty")

    pil_frames: list[Any] = []
    for arr in frames:
        img = Image.fromarray(arr, mode="RGB")
        scale = max(1, int(round(float(frame_display_height) / float(max(1, img.height)))))
        resized = img.resize((img.width * scale, img.height * scale), resample=Image.Resampling.NEAREST)
        pil_frames.append(resized)

    pad = 6
    gap = 4
    caption_height = 36 + max(0, len(subtitle_lines) - 1) * 16
    strip_width = sum(img.width for img in pil_frames) + gap * max(0, len(pil_frames) - 1)
    strip_height = max(img.height for img in pil_frames)
    canvas = Image.new("RGB", (pad * 2 + strip_width, pad * 2 + caption_height + strip_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(16, 24, 40))
    for idx, line in enumerate(subtitle_lines):
        draw.text((pad, pad + 18 + idx * 16), line, fill=(80, 84, 95))

    x = pad
    y = pad + caption_height
    for img in pil_frames:
        canvas.paste(img, (x, y))
        x += img.width + gap
    return canvas


def _write_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int = 2, bg_color: tuple[int, int, int] = (245, 245, 245)) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for contact-sheet rendering. Install via `pip install pillow`.") from exc

    if len(image_paths) == 0:
        return

    images = [Image.open(p).convert("RGB") for p in image_paths]
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    cols = max(1, int(columns))
    rows = (len(images) + cols - 1) // cols
    gap = 8
    sheet = Image.new("RGB", (cols * max_w + (cols + 1) * gap, rows * max_h + (rows + 1) * gap), color=bg_color)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = gap + c * (max_w + gap)
        y = gap + r * (max_h + gap)
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _write_clip_mp4(
    *,
    voxels: np.ndarray,
    out_path: Path,
    split_polarity: bool,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
    tie_epsilon: float,
    fps: float,
) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for MP4 export. Install via `pip install opencv-python`.") from exc

    if voxels.ndim != 4 or voxels.shape[0] <= 0:
        raise ValueError(f"voxels must be [T,C,H,W], got shape={voxels.shape}")

    first_frame = _voxel_to_rgb(
        voxel=np.asarray(voxels[0], dtype=np.float32),
        split_polarity=split_polarity,
        polarity_order=polarity_order,
        vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
        tie_epsilon=tie_epsilon,
    )
    height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(fps, 1e-6)), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")
    try:
        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for t in range(1, voxels.shape[0]):
            frame = _voxel_to_rgb(
                voxel=np.asarray(voxels[t], dtype=np.float32),
                split_polarity=split_polarity,
                polarity_order=polarity_order,
                vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                tie_epsilon=tie_epsilon,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _visualize_selected_examples(
    *,
    example_groups: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    frames_per_clip: int,
    frame_step: int,
    frame_display_height: int,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
    tie_epsilon: float,
    write_mp4: bool,
    mp4_fps: float,
) -> list[dict[str, Any]]:
    _require_h5py()
    written_rows: list[dict[str, Any]] = []
    for group_name, rows in example_groups.items():
        if len(rows) == 0:
            continue
        group_dir = output_dir / group_name
        strip_paths: list[Path] = []
        for example_idx, row in enumerate(rows):
            file_path = Path(str(row["file"]))
            clip_start = int(row["clip_start"])
            with h5py.File(str(file_path), "r") as h5f:
                voxels_ds = h5f["voxels"]
                total_windows = int(voxels_ds.shape[0])
                indices = _clip_indices_from_start(
                    start=clip_start,
                    total_windows=total_windows,
                    frames_per_clip=int(frames_per_clip),
                    frame_step=int(frame_step),
                )
                voxels = _read_voxels_by_indices(voxels_ds, indices)
                channels = int(voxels_ds.shape[1])
                split_polarity = _infer_split_polarity(h5f, channels)
                rgb_frames = [
                    _voxel_to_rgb(
                        voxel=np.asarray(voxels[t], dtype=np.float32),
                        split_polarity=split_polarity,
                        polarity_order=polarity_order,
                        vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                        tie_epsilon=tie_epsilon,
                    )
                    for t in range(voxels.shape[0])
                ]

            title = f"{group_name} | {file_path.name} | start={clip_start}"
            subtitle = [
                f"keep={row['keep']}  margin={_format_float(row.get('filter_margin'))}  reason={row.get('decision_reason', 'n/a')}",
                f"clip_mean_active={_format_float(row.get('clip_mean_active'))}  clip_active_frac={_format_float(row.get('clip_active_frac'))}  clip_mean_score={_format_float(row.get('clip_mean_score'))}",
            ]
            strip_img = _render_clip_strip(
                frames=rgb_frames,
                title=title,
                subtitle_lines=subtitle,
                frame_display_height=int(frame_display_height),
            )
            stem = f"{example_idx:02d}_{file_path.stem}_start{clip_start:06d}"
            strip_path = group_dir / f"{stem}.png"
            group_dir.mkdir(parents=True, exist_ok=True)
            strip_img.save(strip_path)
            strip_paths.append(strip_path)

            mp4_path = None
            if bool(write_mp4):
                mp4_path = group_dir / f"{stem}.mp4"
                _write_clip_mp4(
                    voxels=voxels,
                    out_path=mp4_path,
                    split_polarity=split_polarity,
                    polarity_order=polarity_order,
                    vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                    tie_epsilon=tie_epsilon,
                    fps=float(mp4_fps),
                )

            out_row = dict(row)
            out_row["group_name"] = group_name
            out_row["preview_strip"] = str(strip_path)
            out_row["preview_mp4"] = "" if mp4_path is None else str(mp4_path)
            written_rows.append(out_row)

        _write_contact_sheet(strip_paths, group_dir / "contact_sheet.png", columns=2)
    return written_rows


def main() -> None:
    parser = argparse.ArgumentParser("Visualize which clips are kept or dropped by activity-based filtering.")
    parser.add_argument("--input_path", type=Path, default=None, help="Single voxel H5 file.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory to scan for voxel H5 files.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan --dataset_root for *.h5 files.",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for number of files to analyze.")
    parser.add_argument("--frames_per_clip", type=int, default=16, help="Clip length used for filter simulation.")
    parser.add_argument("--frame_step", type=int, default=1, help="Temporal sampling stride between windows inside a clip.")
    parser.add_argument("--clip_stride", type=int, default=1, help="Stride between candidate clip starts.")
    parser.add_argument("--active_metric", type=str, default=DEFAULT_ACTIVE_METRIC, help="Window-level metric used for activity filtering.")
    parser.add_argument("--score_metric", type=str, default=DEFAULT_SCORE_METRIC, help="Optional secondary metric used for filtering.")
    parser.add_argument("--disable_score_metric", action=argparse.BooleanOptionalAction, default=False, help="Ignore score_metric even if present.")
    parser.add_argument("--active_eps", type=float, default=1e-6, help="Window activity threshold for counting a window as active in clip_active_frac.")
    parser.add_argument("--min_clip_mean_active", type=float, default=0.0, help="Keep clip only if mean(active_metric over clip) >= threshold.")
    parser.add_argument("--min_clip_active_frac", type=float, default=0.0, help="Keep clip only if fraction of active windows in clip >= threshold.")
    parser.add_argument("--min_clip_mean_score", type=float, default=0.0, help="Optional keep threshold for mean(score_metric over clip).")
    parser.add_argument("--hist_bins", type=int, default=100, help="Histogram bins for clip metrics.")
    parser.add_argument("--hist_min", type=float, default=0.0, help="Histogram lower bound.")
    parser.add_argument("--hist_max", type=float, default=1.0, help="Histogram upper bound.")
    parser.add_argument("--num_random_examples", type=int, default=8, help="Random keep/drop examples per bucket.")
    parser.add_argument("--num_boundary_examples", type=int, default=8, help="Near-threshold keep/drop examples per bucket.")
    parser.add_argument("--write_clip_csv", action=argparse.BooleanOptionalAction, default=True, help="Write per-clip decisions to CSV.")
    parser.add_argument("--write_mp4", action=argparse.BooleanOptionalAction, default=False, help="Also export MP4 for selected example clips.")
    parser.add_argument("--mp4_fps", type=float, default=12.0, help="FPS for selected example MP4s.")
    parser.add_argument("--frame_display_height", type=int, default=96, help="Display height for each frame in the preview strip PNGs.")
    parser.add_argument("--polarity_order", choices=["negpos", "posneg"], default="negpos", help="Channel order for split-polarity preview rendering.")
    parser.add_argument("--vote_use_abs_for_split_polarity", action=argparse.BooleanOptionalAction, default=True, help="Use abs(pos) vs abs(neg) in preview vote maps.")
    parser.add_argument("--vote_tie_epsilon", type=float, default=0.0, help="Tie threshold for preview vote maps.")
    parser.add_argument("--chunk_size", type=int, default=131072, help="Chunk size when reading scalar activity metadata.")
    parser.add_argument("--output_dir", type=Path, default=Path("tmp/activity_filter_visualization"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for example sampling.")
    args = parser.parse_args()

    files = _select_files(input_path=args.input_path, dataset_root=args.dataset_root, recursive=bool(args.recursive))
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise FileNotFoundError("no input files found")

    filtered_files = [p for p in files if _is_voxel_h5(p)]
    if len(filtered_files) == 0:
        raise FileNotFoundError("no voxel h5 files found (requires root dataset 'voxels' with 4D shape)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root
    rng = random.Random(int(args.seed))

    use_score_metric = not bool(args.disable_score_metric)
    hist_range = (float(args.hist_min), float(args.hist_max))
    if not hist_range[0] < hist_range[1]:
        raise ValueError("histogram range must satisfy hist_min < hist_max")

    clip_rows_csv = output_dir / "clip_filter_rows.csv"
    clip_writer = None
    clip_handle = None
    if bool(args.write_clip_csv):
        clip_handle = clip_rows_csv.open("w", newline="", encoding="utf-8")
        clip_writer = csv.DictWriter(
            clip_handle,
            fieldnames=[
                "file",
                "relative_file",
                "dataset_family",
                "samples",
                "clip_start",
                "clip_last_index",
                "keep",
                "clip_mean_active",
                "clip_active_frac",
                "clip_mean_score",
                "filter_margin",
                "decision_reason",
            ],
        )
        clip_writer.writeheader()

    per_file_rows: list[dict[str, Any]] = []
    drop_reason_counts: dict[str, int] = {}
    keep_random_examples: list[dict[str, Any]] = []
    drop_random_examples: list[dict[str, Any]] = []
    keep_boundary_examples: list[dict[str, Any]] = []
    drop_boundary_examples: list[dict[str, Any]] = []
    keep_seen = 0
    drop_seen = 0

    hist_payload = {
        "clip_mean_active": {
            "all": np.zeros((int(args.hist_bins),), dtype=np.int64),
            "keep": np.zeros((int(args.hist_bins),), dtype=np.int64),
            "drop": np.zeros((int(args.hist_bins),), dtype=np.int64),
        },
        "clip_active_frac": {
            "all": np.zeros((int(args.hist_bins),), dtype=np.int64),
            "keep": np.zeros((int(args.hist_bins),), dtype=np.int64),
            "drop": np.zeros((int(args.hist_bins),), dtype=np.int64),
        },
    }
    hist_edges = np.linspace(hist_range[0], hist_range[1], int(args.hist_bins) + 1, dtype=np.float64)

    total_clips = 0
    total_keep = 0
    total_drop = 0

    try:
        for file_path in filtered_files:
            with h5py.File(str(file_path), "r") as h5f:
                voxels = h5f["voxels"]
                total_windows = int(voxels.shape[0])
                representation = str(_safe_attr(h5f.attrs, "representation", ""))
                dataset_family = _infer_dataset_family(file_path=file_path, representation=representation)

                if str(args.active_metric) not in h5f:
                    raise KeyError(f"missing dataset '{args.active_metric}' in {file_path}")
                active_values = _read_1d_dataset(h5f[str(args.active_metric)], chunk_size=max(1, int(args.chunk_size)))
                score_values = None
                if use_score_metric:
                    if str(args.score_metric) not in h5f:
                        raise KeyError(f"missing dataset '{args.score_metric}' in {file_path}")
                    score_values = _read_1d_dataset(h5f[str(args.score_metric)], chunk_size=max(1, int(args.chunk_size)))

                starts, clip_mean_active, clip_active_frac, clip_mean_score = _compute_clip_metrics_vectorized(
                    active_values=active_values,
                    score_values=score_values,
                    frames_per_clip=int(args.frames_per_clip),
                    frame_step=int(args.frame_step),
                    clip_stride=int(args.clip_stride),
                    active_eps=float(args.active_eps),
                )

                mean_active_ok = clip_mean_active >= float(args.min_clip_mean_active)
                active_frac_ok = clip_active_frac >= float(args.min_clip_active_frac)
                if use_score_metric:
                    assert clip_mean_score is not None
                    mean_score_ok = clip_mean_score >= float(args.min_clip_mean_score)
                else:
                    mean_score_ok = np.ones_like(mean_active_ok, dtype=bool)
                keep_mask = mean_active_ok & active_frac_ok & mean_score_ok

                margin_active = np.asarray(
                    [_normalized_margin(v, float(args.min_clip_mean_active)) for v in clip_mean_active.tolist()],
                    dtype=np.float32,
                )
                margin_frac = np.asarray(
                    [_normalized_margin(v, float(args.min_clip_active_frac)) for v in clip_active_frac.tolist()],
                    dtype=np.float32,
                )
                if use_score_metric and clip_mean_score is not None:
                    margin_score = np.asarray(
                        [_normalized_margin(v, float(args.min_clip_mean_score)) for v in clip_mean_score.tolist()],
                        dtype=np.float32,
                    )
                    filter_margin = np.minimum(np.minimum(margin_active, margin_frac), margin_score)
                else:
                    filter_margin = np.minimum(margin_active, margin_frac)

                total_clips += int(starts.size)
                keep_count = int(np.count_nonzero(keep_mask))
                drop_count = int(starts.size - keep_count)
                total_keep += keep_count
                total_drop += drop_count

                hist_payload["clip_mean_active"]["all"] += _make_hist_counts(clip_mean_active, bins=int(args.hist_bins), value_range=hist_range)
                hist_payload["clip_mean_active"]["keep"] += _make_hist_counts(clip_mean_active[keep_mask], bins=int(args.hist_bins), value_range=hist_range)
                hist_payload["clip_mean_active"]["drop"] += _make_hist_counts(clip_mean_active[~keep_mask], bins=int(args.hist_bins), value_range=hist_range)
                hist_payload["clip_active_frac"]["all"] += _make_hist_counts(clip_active_frac, bins=int(args.hist_bins), value_range=hist_range)
                hist_payload["clip_active_frac"]["keep"] += _make_hist_counts(clip_active_frac[keep_mask], bins=int(args.hist_bins), value_range=hist_range)
                hist_payload["clip_active_frac"]["drop"] += _make_hist_counts(clip_active_frac[~keep_mask], bins=int(args.hist_bins), value_range=hist_range)

                row_summary = {
                    "file": str(file_path),
                    "relative_file": _relative_or_name(file_path, dataset_root),
                    "dataset_family": dataset_family,
                    "samples": total_windows,
                    "num_candidate_clips": int(starts.size),
                    "num_keep_clips": keep_count,
                    "num_drop_clips": drop_count,
                    "keep_ratio": float(keep_count) / float(max(1, starts.size)),
                    "drop_ratio": float(drop_count) / float(max(1, starts.size)),
                    "clip_mean_active_mean": None if clip_mean_active.size == 0 else float(clip_mean_active.mean()),
                    "clip_active_frac_mean": None if clip_active_frac.size == 0 else float(clip_active_frac.mean()),
                    "clip_mean_score_mean": None if clip_mean_score is None or clip_mean_score.size == 0 else float(clip_mean_score.mean()),
                }
                per_file_rows.append(row_summary)

                for idx in range(int(starts.size)):
                    keep = bool(keep_mask[idx])
                    clip_mean_score_value = None if clip_mean_score is None else float(clip_mean_score[idx])
                    decision_reason = _build_reason_string(
                        mean_active_ok=bool(mean_active_ok[idx]),
                        active_frac_ok=bool(active_frac_ok[idx]),
                        mean_score_ok=bool(mean_score_ok[idx]),
                        use_score=bool(use_score_metric),
                    )
                    row = {
                        "file": str(file_path),
                        "relative_file": _relative_or_name(file_path, dataset_root),
                        "dataset_family": dataset_family,
                        "samples": total_windows,
                        "clip_start": int(starts[idx]),
                        "clip_last_index": int(
                            _clip_indices_from_start(
                                start=int(starts[idx]),
                                total_windows=total_windows,
                                frames_per_clip=int(args.frames_per_clip),
                                frame_step=int(args.frame_step),
                            )[-1]
                        ),
                        "keep": keep,
                        "clip_mean_active": float(clip_mean_active[idx]),
                        "clip_active_frac": float(clip_active_frac[idx]),
                        "clip_mean_score": clip_mean_score_value,
                        "filter_margin": float(filter_margin[idx]),
                        "decision_reason": decision_reason,
                    }
                    if clip_writer is not None:
                        clip_writer.writerow(row)

                    if keep:
                        keep_seen += 1
                        _reservoir_update(
                            keep_random_examples,
                            row,
                            seen_count=keep_seen,
                            max_items=int(args.num_random_examples),
                            rng=rng,
                        )
                        _update_sorted_examples(
                            keep_boundary_examples,
                            row,
                            k=int(args.num_boundary_examples),
                            key_name="filter_margin",
                            reverse=False,
                        )
                    else:
                        drop_reason_counts[decision_reason] = int(drop_reason_counts.get(decision_reason, 0)) + 1
                        drop_seen += 1
                        _reservoir_update(
                            drop_random_examples,
                            row,
                            seen_count=drop_seen,
                            max_items=int(args.num_random_examples),
                            rng=rng,
                        )
                        _update_sorted_examples(
                            drop_boundary_examples,
                            row,
                            k=int(args.num_boundary_examples),
                            key_name="filter_margin",
                            reverse=True,
                        )

            print(
                "[OK] "
                f"{file_path} | clips={int(starts.size)} | keep={keep_count} | drop={drop_count} | "
                f"keep_ratio={row_summary['keep_ratio']:.4f} | drop_ratio={row_summary['drop_ratio']:.4f}"
            )
    finally:
        if clip_handle is not None:
            clip_handle.close()

    if total_clips <= 0:
        raise RuntimeError("no candidate clips were analyzed")

    histogram_metrics: dict[str, dict[str, Any]] = {}
    for metric_name, payload in hist_payload.items():
        all_total = int(payload["all"].sum())
        keep_total = int(payload["keep"].sum())
        drop_total = int(payload["drop"].sum())
        histogram_metrics[metric_name] = {
            "edges": hist_edges,
            "all_frac": payload["all"].astype(np.float64) / float(max(1, all_total)),
            "keep_frac": payload["keep"].astype(np.float64) / float(max(1, keep_total)),
            "drop_frac": payload["drop"].astype(np.float64) / float(max(1, drop_total)),
            "keep_ratio": float(total_keep) / float(max(1, total_clips)),
            "drop_ratio": float(total_drop) / float(max(1, total_clips)),
        }

    keep_random_examples.sort(key=lambda x: (str(x["file"]), int(x["clip_start"])))
    drop_random_examples.sort(key=lambda x: (str(x["file"]), int(x["clip_start"])))

    example_groups = {
        "keep_random": keep_random_examples,
        "drop_random": drop_random_examples,
        "keep_boundary": keep_boundary_examples,
        "drop_boundary": drop_boundary_examples,
    }
    example_rows = _visualize_selected_examples(
        example_groups=example_groups,
        output_dir=output_dir / "examples",
        frames_per_clip=int(args.frames_per_clip),
        frame_step=int(args.frame_step),
        frame_display_height=int(args.frame_display_height),
        polarity_order=str(args.polarity_order),
        vote_use_abs_for_split_polarity=bool(args.vote_use_abs_for_split_polarity),
        tie_epsilon=float(args.vote_tie_epsilon),
        write_mp4=bool(args.write_mp4),
        mp4_fps=float(args.mp4_fps),
    )

    summary = {
        "input_path": None if args.input_path is None else str(args.input_path),
        "dataset_root": None if dataset_root is None else str(dataset_root),
        "num_files": len(filtered_files),
        "frames_per_clip": int(args.frames_per_clip),
        "frame_step": int(args.frame_step),
        "clip_stride": int(args.clip_stride),
        "active_metric": str(args.active_metric),
        "score_metric": None if not use_score_metric else str(args.score_metric),
        "active_eps": float(args.active_eps),
        "min_clip_mean_active": float(args.min_clip_mean_active),
        "min_clip_active_frac": float(args.min_clip_active_frac),
        "min_clip_mean_score": None if not use_score_metric else float(args.min_clip_mean_score),
        "num_candidate_clips": int(total_clips),
        "num_keep_clips": int(total_keep),
        "num_drop_clips": int(total_drop),
        "keep_ratio": float(total_keep) / float(max(1, total_clips)),
        "drop_ratio": float(total_drop) / float(max(1, total_clips)),
        "drop_reason_counts": drop_reason_counts,
    }

    summary_json = output_dir / "clip_filter_summary.json"
    per_file_csv = output_dir / "clip_filter_per_file.csv"
    examples_csv = output_dir / "clip_filter_examples.csv"
    histogram_svg = output_dir / "clip_filter_histograms.svg"
    histogram_csv = output_dir / "clip_filter_histogram_bins.csv"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(
        path=per_file_csv,
        rows=per_file_rows,
        columns=[
            "file",
            "relative_file",
            "dataset_family",
            "samples",
            "num_candidate_clips",
            "num_keep_clips",
            "num_drop_clips",
            "keep_ratio",
            "drop_ratio",
            "clip_mean_active_mean",
            "clip_active_frac_mean",
            "clip_mean_score_mean",
        ],
    )
    _write_csv(
        path=examples_csv,
        rows=example_rows,
        columns=[
            "group_name",
            "file",
            "relative_file",
            "dataset_family",
            "clip_start",
            "clip_last_index",
            "keep",
            "clip_mean_active",
            "clip_active_frac",
            "clip_mean_score",
            "filter_margin",
            "decision_reason",
            "preview_strip",
            "preview_mp4",
        ],
    )

    histogram_rows: list[dict[str, Any]] = []
    for metric_name, payload in histogram_metrics.items():
        edges = payload["edges"]
        all_frac = payload["all_frac"]
        keep_frac = payload["keep_frac"]
        drop_frac = payload["drop_frac"]
        for idx in range(len(all_frac)):
            histogram_rows.append(
                {
                    "metric": metric_name,
                    "bin_index": idx,
                    "bin_start": float(edges[idx]),
                    "bin_end": float(edges[idx + 1]),
                    "all_fraction": float(all_frac[idx]),
                    "keep_fraction": float(keep_frac[idx]),
                    "drop_fraction": float(drop_frac[idx]),
                }
            )
    _write_csv(
        path=histogram_csv,
        rows=histogram_rows,
        columns=["metric", "bin_index", "bin_start", "bin_end", "all_fraction", "keep_fraction", "drop_fraction"],
    )
    _build_keep_drop_histogram_svg(
        output_path=histogram_svg,
        metrics=histogram_metrics,
        value_range=hist_range,
        chart_title="Clip Filter Keep/Drop Histograms",
    )

    print(
        "[SUMMARY] "
        f"files={len(filtered_files)} | clips={int(total_clips)} | keep={int(total_keep)} | drop={int(total_drop)} | "
        f"summary_json={summary_json} | per_file_csv={per_file_csv} | examples_csv={examples_csv} | "
        f"histogram_svg={histogram_svg} | clip_rows_csv={clip_rows_csv if bool(args.write_clip_csv) else 'disabled'}"
    )


if __name__ == "__main__":
    main()
