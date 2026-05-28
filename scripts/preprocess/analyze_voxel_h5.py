from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  
import numpy as np


def _safe_attr(attrs: h5py.AttributeManager, key: str, default: Any = None) -> Any:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_voxel_h5(path: Path) -> bool:
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


def _start_end_us(h5f: h5py.File, n_samples: int) -> tuple[int | None, int | None]:
    if n_samples <= 0:
        return None, None

    if "window_t_start_us" in h5f and "window_t_end_us" in h5f:
        starts = h5f["window_t_start_us"]
        ends = h5f["window_t_end_us"]
        if len(starts) >= n_samples and len(ends) >= n_samples:
            return int(starts[0]), int(ends[n_samples - 1])

    if "anchor_timestamp_us" in h5f:
        anchors = h5f["anchor_timestamp_us"]
        if len(anchors) >= n_samples:
            return int(anchors[0]), int(anchors[n_samples - 1])

    origin = _safe_attr(h5f.attrs, "time_origin_us", None)
    if (
        origin is not None
        and int(origin) >= 0
        and "window_rel_start_us" in h5f
        and "window_rel_end_us" in h5f
    ):
        rel_starts = h5f["window_rel_start_us"]
        rel_ends = h5f["window_rel_end_us"]
        if len(rel_starts) >= n_samples and len(rel_ends) >= n_samples:
            return int(origin) + int(rel_starts[0]), int(origin) + int(rel_ends[n_samples - 1])

    return None, None


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

        # Majority vote per temporal bin:
        #   vote_bin > 0: positive wins this bin
        #   vote_bin < 0: negative wins this bin
        #   vote_bin = 0: tie/no-event in this bin
        vote_bins = np.sign(pos_bins - neg_bins)
        vote_score = vote_bins.sum(axis=0)
    else:
        # Fallback for non-split polarity representations.
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


def _pick_indices(n_samples: int, num_visualizations: int, explicit_indices: list[int] | None) -> list[int]:
    if n_samples <= 0:
        return []

    if explicit_indices is not None and len(explicit_indices) > 0:
        out = sorted(set([int(i) for i in explicit_indices if 0 <= int(i) < n_samples]))
        return out

    k = min(max(0, int(num_visualizations)), n_samples)
    if k == 0:
        return []
    if k == 1:
        return [0]
    return sorted(set(np.linspace(0, n_samples - 1, num=k, dtype=int).tolist()))


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _format_optional_float(value: float | None, fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(float(value), fmt)


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


def _build_compare_oneline(result: dict[str, Any]) -> str:
    mean_ev = result.get("window_event_count_mean")
    nz_ratio = result.get("voxel_nonzero_ratio")
    zero_windows = result.get("num_zero_event_windows")
    samples = int(result.get("samples", 0))
    dataset = str(result.get("dataset_family", "other"))

    mean_ev_txt = _format_optional_float(None if mean_ev is None else float(mean_ev), ".2f")
    nz_ratio_txt = _format_optional_float(None if nz_ratio is None else float(nz_ratio), ".6f")
    if zero_windows is None:
        zero_txt = "n/a"
    else:
        zero_txt = f"{int(zero_windows)}/{samples}"
    return (
        f"dataset={dataset} | voxel_nonzero_ratio={nz_ratio_txt} | "
        f"mean_event_count={mean_ev_txt} | zero_event_windows={zero_txt}"
    )


def _write_preview_images(
    *,
    h5f: h5py.File,
    file_path: Path,
    output_dir: Path,
    dataset_root: Path | None,
    n_samples: int,
    num_visualizations: int,
    explicit_indices: list[int] | None,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
    tie_epsilon: float,
) -> list[str]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for visualization. Install via `pip install pillow`.") from exc

    voxels = h5f["voxels"]
    channels = int(voxels.shape[1])
    split_polarity = _infer_split_polarity(h5f, channels)
    indices = _pick_indices(
        n_samples=n_samples,
        num_visualizations=num_visualizations,
        explicit_indices=explicit_indices,
    )
    if len(indices) == 0:
        return []

    relative_name = _relative_or_name(file_path, dataset_root).replace("/", "__")
    relative_name = relative_name.replace("\\", "__")
    stem = Path(relative_name).with_suffix("").name
    preview_dir = output_dir / "previews" / stem
    preview_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[str] = []
    for idx in indices:
        voxel = np.asarray(voxels[idx], dtype=np.float32)
        rgb = _voxel_to_rgb(
            voxel=voxel,
            split_polarity=split_polarity,
            polarity_order=polarity_order,
            vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
            tie_epsilon=tie_epsilon,
        )
        out_path = preview_dir / f"window_{idx:06d}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path)
        written_paths.append(str(out_path))
    return written_paths


def _write_preview_video(
    *,
    h5f: h5py.File,
    file_path: Path,
    output_dir: Path,
    dataset_root: Path | None,
    n_samples: int,
    explicit_indices: list[int] | None,
    num_visualizations: int,
    use_all_windows: bool,
    max_frames: int | None,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
    tie_epsilon: float,
    fps: float,
) -> tuple[str | None, int]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for MP4 export. Install via `pip install opencv-python`.") from exc

    if use_all_windows:
        indices = list(range(n_samples))
    else:
        indices = _pick_indices(
            n_samples=n_samples,
            num_visualizations=num_visualizations,
            explicit_indices=explicit_indices,
        )
    if len(indices) == 0:
        return None, 0

    if max_frames is not None and int(max_frames) > 0 and len(indices) > int(max_frames):
        indices = indices[: int(max_frames)]

    voxels = h5f["voxels"]
    channels = int(voxels.shape[1])
    split_polarity = _infer_split_polarity(h5f, channels)

    relative_name = _relative_or_name(file_path, dataset_root).replace("/", "__")
    relative_name = relative_name.replace("\\", "__")
    stem = Path(relative_name).with_suffix("").name
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"{stem}.mp4"

    first_frame = _voxel_to_rgb(
        voxel=np.asarray(voxels[indices[0]], dtype=np.float32),
        split_polarity=split_polarity,
        polarity_order=polarity_order,
        vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
        tie_epsilon=tie_epsilon,
    )
    height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(fps, 1e-6)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")

    try:
        writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
        for idx in indices[1:]:
            frame = _voxel_to_rgb(
                voxel=np.asarray(voxels[idx], dtype=np.float32),
                split_polarity=split_polarity,
                polarity_order=polarity_order,
                vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                tie_epsilon=tie_epsilon,
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return str(out_path), len(indices)


def _analyze_file(
    *,
    file_path: Path,
    output_dir: Path,
    dataset_root: Path | None,
    metadata_only: bool,
    with_visualization: bool,
    num_visualizations: int,
    explicit_indices: list[int] | None,
    polarity_order: str,
    vote_use_abs_for_split_polarity: bool,
    tie_epsilon: float,
    write_mp4: bool,
    mp4_fps: float,
    mp4_use_all_windows: bool,
    mp4_max_frames: int | None,
) -> dict[str, Any]:
    with h5py.File(str(file_path), "r") as h5f:
        if "voxels" not in h5f:
            raise ValueError("missing dataset 'voxels'")
        voxels = h5f["voxels"]
        if voxels.ndim != 4:
            raise ValueError(f"voxels must be 4D, got shape={voxels.shape}")

        n_samples, channels, height, width = [int(v) for v in voxels.shape]
        start_us, end_us = _start_end_us(h5f=h5f, n_samples=n_samples)
        duration_us = None if start_us is None or end_us is None else max(0, int(end_us) - int(start_us))
        duration_s = None if duration_us is None else float(duration_us) / 1e6
        windows_per_second = None
        if duration_s is not None and duration_s > 0:
            windows_per_second = float(n_samples) / float(duration_s)

        preview_paths: list[str] = []
        preview_video_path: str | None = None
        preview_video_num_frames = 0
        if with_visualization:
            preview_paths = _write_preview_images(
                h5f=h5f,
                file_path=file_path,
                output_dir=output_dir,
                dataset_root=dataset_root,
                n_samples=n_samples,
                num_visualizations=num_visualizations,
                explicit_indices=explicit_indices,
                polarity_order=polarity_order,
                vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                tie_epsilon=tie_epsilon,
            )
        if write_mp4:
            preview_video_path, preview_video_num_frames = _write_preview_video(
                h5f=h5f,
                file_path=file_path,
                output_dir=output_dir,
                dataset_root=dataset_root,
                n_samples=n_samples,
                explicit_indices=explicit_indices,
                num_visualizations=num_visualizations,
                use_all_windows=mp4_use_all_windows,
                max_frames=mp4_max_frames,
                polarity_order=polarity_order,
                vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                tie_epsilon=tie_epsilon,
                fps=mp4_fps,
            )

        # Debug stats for event/window coverage and voxel value distribution.
        window_event_count_total = None
        window_event_count_mean = None
        window_event_count_min = None
        window_event_count_max = None
        num_zero_event_windows = None
        if "window_event_count" in h5f and len(h5f["window_event_count"]) >= n_samples:
            event_counts = np.asarray(h5f["window_event_count"][:n_samples], dtype=np.int64)
            if event_counts.size > 0:
                window_event_count_total = int(event_counts.sum())
                window_event_count_mean = float(event_counts.mean())
                window_event_count_min = int(event_counts.min())
                window_event_count_max = int(event_counts.max())
                num_zero_event_windows = int(np.count_nonzero(event_counts == 0))
            else:
                window_event_count_total = 0
                window_event_count_mean = 0.0
                window_event_count_min = 0
                window_event_count_max = 0
                num_zero_event_windows = 0

        voxel_global_min = None
        voxel_global_max = None
        voxel_global_mean = None
        voxel_global_std = None
        voxel_nonzero_ratio = None
        nonempty_windows = None
        debug_vote_stats: list[dict[str, Any]] = []
        if not metadata_only:
            nonempty_windows = 0
            total_values = 0
            total_nonzero = 0
            total_sum = 0.0
            total_sqsum = 0.0
            chunk_size = 16
            for s in range(0, n_samples, chunk_size):
                e = min(n_samples, s + chunk_size)
                arr = np.asarray(voxels[s:e], dtype=np.float32)
                if arr.size == 0:
                    continue
                arr_min = float(arr.min())
                arr_max = float(arr.max())
                voxel_global_min = arr_min if voxel_global_min is None else min(voxel_global_min, arr_min)
                voxel_global_max = arr_max if voxel_global_max is None else max(voxel_global_max, arr_max)
                total_values += int(arr.size)
                total_nonzero += int(np.count_nonzero(arr != 0))
                total_sum += float(arr.sum())
                total_sqsum += float(np.square(arr).sum())
                nonempty_windows += int(np.count_nonzero(np.max(np.abs(arr), axis=(1, 2, 3)) > 0))
            if total_values > 0:
                voxel_global_mean = total_sum / float(total_values)
                variance = max(total_sqsum / float(total_values) - voxel_global_mean * voxel_global_mean, 0.0)
                voxel_global_std = math.sqrt(variance)
                voxel_nonzero_ratio = float(total_nonzero) / float(total_values)

            # Vote-map diagnostics for selected windows.
            debug_indices = _pick_indices(
                n_samples=n_samples,
                num_visualizations=max(1, int(num_visualizations)),
                explicit_indices=explicit_indices,
            )
            eps = float(max(0.0, tie_epsilon))
            split_polarity = _infer_split_polarity(h5f, channels)
            for idx in debug_indices:
                voxel = np.asarray(voxels[idx], dtype=np.float32)
                vote_score = _vote_score_map(
                    voxel=voxel,
                    split_polarity=split_polarity,
                    polarity_order=polarity_order,
                    vote_use_abs_for_split_polarity=vote_use_abs_for_split_polarity,
                )
                num_pos = int(np.count_nonzero(vote_score > eps))
                num_neg = int(np.count_nonzero(vote_score < -eps))
                num_bg = int(vote_score.size - num_pos - num_neg)
                debug_vote_stats.append(
                    {
                        "index": int(idx),
                        "vote_pos_pixels": num_pos,
                        "vote_neg_pixels": num_neg,
                        "vote_bg_pixels": num_bg,
                        "vote_min": float(vote_score.min()) if vote_score.size > 0 else 0.0,
                        "vote_max": float(vote_score.max()) if vote_score.size > 0 else 0.0,
                        "voxel_min": float(voxel.min()) if voxel.size > 0 else 0.0,
                        "voxel_max": float(voxel.max()) if voxel.size > 0 else 0.0,
                        "voxel_mean": float(voxel.mean()) if voxel.size > 0 else 0.0,
                    }
                )

        representation = _safe_attr(h5f.attrs, "representation", "")
        dataset_family = _infer_dataset_family(file_path=file_path, representation=str(representation))
        result = {
            "file": str(file_path),
            "file_size_mb": round(float(file_path.stat().st_size) / (1024.0 * 1024.0), 3),
            "representation": representation,
            "dataset_family": dataset_family,
            "dtype": str(voxels.dtype),
            "shape": [n_samples, channels, height, width],
            "samples": n_samples,
            "channels": channels,
            "height": height,
            "width": width,
            "t_bins": _safe_attr(h5f.attrs, "t_bins", None),
            "split_polarity": _infer_split_polarity(h5f, channels),
            "window_mode": _safe_attr(h5f.attrs, "window_mode", ""),
            "recording_start_us": start_us,
            "recording_end_us": end_us,
            "estimated_recording_duration_s": duration_s,
            "estimated_windows_per_second": windows_per_second,
            "window_event_count_total": window_event_count_total,
            "window_event_count_mean": window_event_count_mean,
            "window_event_count_min": window_event_count_min,
            "window_event_count_max": window_event_count_max,
            "num_zero_event_windows": num_zero_event_windows,
            "num_nonempty_voxel_windows": None if nonempty_windows is None else int(nonempty_windows),
            "voxel_global_min": voxel_global_min,
            "voxel_global_max": voxel_global_max,
            "voxel_global_mean": voxel_global_mean,
            "voxel_global_std": voxel_global_std,
            "voxel_nonzero_ratio": voxel_nonzero_ratio,
            "metadata_only": bool(metadata_only),
            "visualization_polarity_order": polarity_order,
            "visualization_vote_use_abs_for_split_polarity": bool(vote_use_abs_for_split_polarity),
            "visualization_tie_epsilon": float(tie_epsilon),
            "vote_debug_windows": debug_vote_stats,
            "preview_images": preview_paths,
            "preview_video": preview_video_path,
            "preview_video_num_frames": preview_video_num_frames,
        }
        result["compare_oneline"] = _build_compare_oneline(result)
        return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "file",
        "dataset_family",
        "file_size_mb",
        "representation",
        "dtype",
        "samples",
        "channels",
        "height",
        "width",
        "t_bins",
        "split_polarity",
        "window_mode",
        "recording_start_us",
        "recording_end_us",
        "estimated_recording_duration_s",
        "estimated_windows_per_second",
        "window_event_count_total",
        "window_event_count_mean",
        "window_event_count_min",
        "window_event_count_max",
        "num_zero_event_windows",
        "num_nonempty_voxel_windows",
        "voxel_global_min",
        "voxel_global_max",
        "voxel_global_mean",
        "voxel_global_std",
        "voxel_nonzero_ratio",
        "compare_oneline",
        "preview_images",
        "preview_video",
        "preview_video_num_frames",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in columns}
            out["preview_images"] = ";".join(row.get("preview_images", []))
            writer.writerow(out)


def _print_compare_summary(results: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        family = str(row.get("dataset_family", "other"))
        grouped.setdefault(family, []).append(row)

    for family in sorted(grouped.keys()):
        rows = grouped[family]
        files = len(rows)
        total_samples = sum(int(r.get("samples", 0)) for r in rows)

        total_event_count = 0
        total_event_windows = 0
        for r in rows:
            total = r.get("window_event_count_total")
            n = int(r.get("samples", 0))
            if total is None or n <= 0:
                continue
            total_event_count += int(total)
            total_event_windows += n
        mean_event_count = (
            float(total_event_count) / float(total_event_windows) if total_event_windows > 0 else None
        )

        weighted_nonzero_sum = 0.0
        weighted_nonzero_denom = 0
        for r in rows:
            nz = r.get("voxel_nonzero_ratio")
            if nz is None:
                continue
            n = int(r.get("samples", 0))
            c = int(r.get("channels", 0))
            h = int(r.get("height", 0))
            w = int(r.get("width", 0))
            weight = n * c * h * w
            if weight <= 0:
                continue
            weighted_nonzero_sum += float(nz) * float(weight)
            weighted_nonzero_denom += weight
        voxel_nonzero_ratio = (
            weighted_nonzero_sum / float(weighted_nonzero_denom) if weighted_nonzero_denom > 0 else None
        )

        mean_event_txt = _format_optional_float(mean_event_count, ".2f")
        nz_txt = _format_optional_float(voxel_nonzero_ratio, ".6f")
        print(
            "[COMPARE] "
            f"dataset={family} | files={files} | samples={total_samples} | "
            f"mean_event_count={mean_event_txt} | voxel_nonzero_ratio={nz_txt}"
        )


def main() -> None:
    parser = argparse.ArgumentParser("Analyze preprocessed voxel H5 files and save quick RGB previews.")
    parser.add_argument("--input_path", type=Path, default=None, help="Single voxel H5 file.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory to scan for voxel H5 files.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan --dataset_root for *.h5 files.",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for number of files to analyze.")
    parser.add_argument("--output_dir", type=Path, default=Path("tmp/voxel_h5_analysis"), help="Output directory.")
    parser.add_argument(
        "--metadata_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fast mode: skip heavy voxel-value scans and vote-map diagnostics. Keep timestamp/sample debug.",
    )
    parser.add_argument(
        "--with_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save quick RGB preview images from voxel windows.",
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=3,
        help="Number of preview windows per file when --visualization_indices is not set.",
    )
    parser.add_argument(
        "--visualization_indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit window indices to visualize (e.g. 0 10 100).",
    )
    parser.add_argument(
        "--polarity_order",
        choices=["negpos", "posneg"],
        default="negpos",
        help="Channel order for split polarity visualization: negpos means first-half=neg, second-half=pos.",
    )
    parser.add_argument(
        "--vote_use_abs_for_split_polarity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For split-polarity voxels, compare abs(pos_bin) vs abs(neg_bin) per bin for majority vote.",
    )
    parser.add_argument(
        "--vote_tie_epsilon",
        type=float,
        default=0.0,
        help="Tie threshold for vote score. abs(score)<=epsilon is background.",
    )
    parser.add_argument(
        "--write_mp4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write MP4 preview per analyzed H5.",
    )
    parser.add_argument("--mp4_fps", type=float, default=20.0, help="FPS for MP4 preview.")
    parser.add_argument(
        "--mp4_use_all_windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all windows for MP4 (default: true).",
    )
    parser.add_argument(
        "--mp4_max_frames",
        type=int,
        default=None,
        help="Optional cap on frames per MP4 (useful for very long sequences).",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <output_dir>/summary.json).",
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=None,
        help="Optional CSV report path (default: <output_dir>/summary.csv).",
    )
    args = parser.parse_args()

    with_visualization = bool(args.with_visualization)
    write_mp4 = bool(args.write_mp4)
    if bool(args.metadata_only):
        with_visualization = False
        write_mp4 = False

    files = _select_files(
        input_path=args.input_path,
        dataset_root=args.dataset_root,
        recursive=bool(args.recursive),
    )
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise FileNotFoundError("no input files found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_files: list[Path] = []
    for file_path in files:
        if _is_voxel_h5(file_path):
            filtered_files.append(file_path)
    if len(filtered_files) == 0:
        raise FileNotFoundError("no voxel h5 files found (requires root dataset 'voxels' with 4D shape)")

    results: list[dict[str, Any]] = []
    for file_path in filtered_files:
        try:
            result = _analyze_file(
                file_path=file_path,
                output_dir=output_dir,
                dataset_root=args.dataset_root,
                metadata_only=bool(args.metadata_only),
                with_visualization=with_visualization,
                num_visualizations=int(args.num_visualizations),
                explicit_indices=args.visualization_indices,
                polarity_order=str(args.polarity_order),
                vote_use_abs_for_split_polarity=bool(args.vote_use_abs_for_split_polarity),
                tie_epsilon=float(args.vote_tie_epsilon),
                write_mp4=write_mp4,
                mp4_fps=float(args.mp4_fps),
                mp4_use_all_windows=bool(args.mp4_use_all_windows),
                mp4_max_frames=None if args.mp4_max_frames is None else int(args.mp4_max_frames),
            )
            results.append(result)
            duration_s = result["estimated_recording_duration_s"]
            duration_txt = "n/a" if duration_s is None else f"{duration_s:.3f}s"
            mp4_txt = (
                f"{result['preview_video_num_frames']}f"
                if result.get("preview_video") is not None
                else "off"
            )
            zero_ev_txt = (
                "n/a"
                if result["num_zero_event_windows"] is None
                else f"{result['num_zero_event_windows']}/{result['samples']}"
            )
            print(
                "[OK] "
                f"{file_path} | samples={result['samples']} | "
                f"shape={tuple(result['shape'])} | duration={duration_txt} | "
                f"zero_event_windows={zero_ev_txt} | compare={result['compare_oneline']} | mp4={mp4_txt}"
            )
        except Exception as exc:
            print(f"[FAILED] {file_path}: {exc}")

    if len(results) == 0:
        raise RuntimeError("all files failed to analyze")

    report_json = args.report_json if args.report_json is not None else (output_dir / "summary.json")
    report_csv = args.report_csv if args.report_csv is not None else (output_dir / "summary.csv")
    report_json.parent.mkdir(parents=True, exist_ok=True)

    with report_json.open("w") as f:
        json.dump(results, f, indent=2)
    _write_csv(path=report_csv, rows=results)

    total_samples = sum(int(r["samples"]) for r in results)
    durations = [r["estimated_recording_duration_s"] for r in results if r["estimated_recording_duration_s"] is not None]
    total_duration = float(sum(durations)) if len(durations) > 0 else None
    total_duration_txt = "n/a" if total_duration is None else f"{total_duration:.3f}s"
    print(
        "[SUMMARY] "
        f"files={len(results)} | total_samples={total_samples} | "
        f"total_estimated_duration={total_duration_txt} | "
        f"json={report_json} | csv={report_csv}"
    )
    _print_compare_summary(results)


if __name__ == "__main__":
    main()
