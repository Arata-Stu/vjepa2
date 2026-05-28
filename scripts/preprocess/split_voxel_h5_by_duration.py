from __future__ import annotations

import argparse
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time

import h5py
import hdf5plugin  # noqa: F401 (registers hdf5 plugins)
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess.utils import (
    RgbMp4Writer,
    cleanup_tmp_file,
    get_h5_compression_flags,
    tmp_media_output_path,
    tmp_output_path,
)


H5_COMPRESSION_FLAGS = get_h5_compression_flags()
_PROGRESS_QUEUE = None


@dataclass(frozen=True)
class CompanionMp4Info:
    source_path: Path
    fps: float
    fps_source: str


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            return "voxels" in h5f and h5f["voxels"].ndim == 4 and h5f["voxels"].shape[0] > 0
    except Exception:
        return False


def _select_input_files(
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
        files = sorted([p for p in iterator if p.is_file()])

    files = [p for p in files if _is_voxel_h5(p)]
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def _resolve_output_path(
    input_path: Path,
    dataset_root: Path | None,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return input_path.parent / f"{input_path.stem}_split" / input_path.name

    if dataset_root is None:
        return output_root / input_path.name

    rel = input_path.relative_to(dataset_root)
    return output_root / rel


def _anchor_times_us(h5f: h5py.File, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0,), dtype=np.int64)

    if "anchor_timestamp_us" in h5f and len(h5f["anchor_timestamp_us"]) >= n_samples:
        return np.asarray(h5f["anchor_timestamp_us"][:n_samples], dtype=np.int64)

    if "window_t_start_us" in h5f and "window_t_end_us" in h5f:
        starts = np.asarray(h5f["window_t_start_us"][:n_samples], dtype=np.int64)
        ends = np.asarray(h5f["window_t_end_us"][:n_samples], dtype=np.int64)
        return starts + ((ends - starts) // 2)

    if "window_rel_start_us" in h5f and "window_rel_end_us" in h5f and "time_origin_us" in h5f.attrs:
        origin = int(h5f.attrs["time_origin_us"])
        rel_starts = np.asarray(h5f["window_rel_start_us"][:n_samples], dtype=np.int64)
        rel_ends = np.asarray(h5f["window_rel_end_us"][:n_samples], dtype=np.int64)
        return origin + rel_starts + ((rel_ends - rel_starts) // 2)

    raise KeyError("cannot resolve anchor timestamps from H5 (missing anchor/window time datasets)")


def _contiguous_chunk_ranges(anchors_us: np.ndarray, chunk_duration_us: int) -> list[tuple[int, int]]:
    if anchors_us.size == 0:
        return []
    if int(chunk_duration_us) <= 0:
        raise ValueError("chunk_duration_us must be > 0")
    if anchors_us.size > 1 and np.any(anchors_us[1:] < anchors_us[:-1]):
        raise ValueError("anchor timestamps must be non-decreasing")

    first = int(anchors_us[0])
    chunk_ids = ((anchors_us - first) // int(chunk_duration_us)).astype(np.int64, copy=False)
    change = np.where(chunk_ids[1:] != chunk_ids[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [anchors_us.size]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _dataset_create_kwargs(src_ds: h5py.Dataset, out_shape: tuple[int, ...]) -> dict:
    if len(out_shape) == 0:
        return {}

    kwargs: dict = {}
    vlen_dtype = h5py.check_dtype(vlen=src_ds.dtype)
    is_ref_dtype = h5py.check_dtype(ref=src_ds.dtype) is not None
    is_string_like = src_ds.dtype.kind in {"S", "U", "O"} or vlen_dtype is not None
    dtype_allows_filter = (not is_string_like) and (not is_ref_dtype)

    if src_ds.ndim > 0 and src_ds.shape[0] > 0:
        if src_ds.chunks is not None and not is_string_like:
            chunks = list(src_ds.chunks)
            chunks[0] = min(max(1, chunks[0]), max(1, out_shape[0]))
            for i in range(1, min(len(chunks), len(out_shape))):
                chunks[i] = min(max(1, chunks[i]), max(1, out_shape[i]))
            kwargs["chunks"] = tuple(chunks[: len(out_shape)])
        elif not is_string_like:
            kwargs["chunks"] = True

    # Keep compression on heavy numeric tensors, but avoid applying plugin filters
    # to variable-length/object/string/reference dtypes to prevent HDF5 plugin crashes.
    if dtype_allows_filter and src_ds.name.endswith("/voxels"):
        kwargs.update(H5_COMPRESSION_FLAGS)
    return kwargs


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_h5_string(value.item())
        if value.size == 1:
            return _decode_h5_string(value.reshape(-1)[0])
    return str(value)


def _load_h5_attr_str(h5f: h5py.File, key: str, default: str = "") -> str:
    if key not in h5f.attrs:
        return default
    return _decode_h5_string(h5f.attrs[key]).strip()


def _safe_attr(attrs: h5py.AttributeManager, key: str, default=None):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_companion_mp4_info(src_h5: h5py.File, input_path: Path) -> CompanionMp4Info | None:
    has_companion = bool(int(_safe_attr(src_h5.attrs, "has_companion_mp4", 0) or 0))
    relpath = _load_h5_attr_str(src_h5, "companion_mp4_relpath", default="")
    if not has_companion and len(relpath) == 0:
        return None

    source_path = Path(relpath) if len(relpath) > 0 else input_path.with_suffix(".mp4")
    if not source_path.is_absolute():
        source_path = input_path.parent / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"companion mp4 not found for {input_path}: {source_path}")

    fps = float(_safe_attr(src_h5.attrs, "companion_mp4_fps", 0.0) or 0.0)
    fps_source = _load_h5_attr_str(src_h5, "companion_mp4_fps_source", default="")
    return CompanionMp4Info(
        source_path=source_path,
        fps=fps,
        fps_source=fps_source,
    )


def _read_video_fps(source_path: Path) -> float:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for companion MP4 splitting.") from exc

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open companion mp4 for reading: {source_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return fps


def _resolved_companion_mp4_fps(info: CompanionMp4Info) -> tuple[float, str]:
    if float(info.fps) > 0.0:
        return float(info.fps), (info.fps_source or "source_h5_attr")
    detected_fps = _read_video_fps(info.source_path)
    if detected_fps > 0.0:
        return float(detected_fps), "source_mp4_detected"
    raise RuntimeError(f"could not determine FPS for companion mp4: {info.source_path}")


def _chunk_h5_has_matching_companion(out_h5_path: Path, out_mp4_path: Path) -> bool:
    if not out_h5_path.exists():
        return False
    try:
        with h5py.File(str(out_h5_path), "r") as h5f:
            has_companion = bool(int(_safe_attr(h5f.attrs, "has_companion_mp4", 0) or 0))
            relpath = _load_h5_attr_str(h5f, "companion_mp4_relpath", default="")
            return has_companion and relpath == out_mp4_path.name
    except Exception:
        return False


def _update_chunk_companion_mp4_attrs(
    *,
    out_h5_path: Path,
    out_mp4_path: Path,
    fps: float,
    fps_source: str,
) -> None:
    with h5py.File(str(out_h5_path), "r+") as out_h5:
        out_h5.attrs["has_companion_mp4"] = 1
        out_h5.attrs["companion_mp4_relpath"] = out_mp4_path.name
        out_h5.attrs["companion_mp4_fps"] = float(fps)
        out_h5.attrs["companion_mp4_fps_source"] = str(fps_source)


def _write_chunk_companion_mp4(
    *,
    source_mp4_path: Path,
    out_mp4_path: Path,
    src_start: int,
    src_end: int,
    fps: float,
) -> None:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for companion MP4 splitting.") from exc

    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4_path = tmp_media_output_path(out_mp4_path, ".tmp")
    cleanup_tmp_file(tmp_mp4_path, context=f"start chunk MP4 write {out_mp4_path}", strict=True)

    cap = cv2.VideoCapture(str(source_mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open companion mp4 for reading: {source_mp4_path}")

    writer: RgbMp4Writer | None = None
    frames_written = 0
    expected_frames = int(src_end - src_start)
    try:
        if int(src_start) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(src_start))

        for _ in range(expected_frames):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(
                    f"unexpected end of companion mp4 while reading "
                    f"frames [{src_start}:{src_end}) from {source_mp4_path}"
                )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if writer is None:
                writer = RgbMp4Writer(
                    tmp_mp4_path,
                    fps=float(fps),
                    width=int(frame_rgb.shape[1]),
                    height=int(frame_rgb.shape[0]),
                )
            writer.write_rgb(frame_rgb)
            frames_written += 1

        if frames_written != expected_frames:
            raise RuntimeError(
                f"chunk mp4 frame count mismatch for {out_mp4_path}: "
                f"expected {expected_frames}, wrote {frames_written}"
            )
    except Exception:
        if writer is not None:
            writer.close()
        cleanup_tmp_file(tmp_mp4_path, context=f"exception chunk MP4 cleanup {out_mp4_path}", strict=False)
        raise
    finally:
        cap.release()

    assert writer is not None
    writer.close()
    os.replace(tmp_mp4_path, out_mp4_path)


def _progress_write(message: str) -> None:
    tqdm.tqdm.write(message)


def _init_worker_progress_queue(progress_queue) -> None:
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue


def _emit_progress(message: str) -> None:
    if _PROGRESS_QUEUE is None:
        _progress_write(message)
        return
    try:
        _PROGRESS_QUEUE.put(message)
    except Exception:
        _progress_write(message)


def _progress_consumer_loop(progress_queue) -> None:
    while True:
        try:
            message = progress_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        if message is None:
            break
        _progress_write(str(message))


def _copy_dataset_slice(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    src_start: int,
    src_end: int,
    copy_batch_size: int,
    progress_interval_s: float = 0.0,
    progress_context: str | None = None,
) -> None:
    if src_ds.ndim == 0:
        dst_ds[()] = src_ds[()]
        return
    if src_end <= src_start:
        return

    copy_batch_size = max(1, int(copy_batch_size))
    total_rows = int(src_end - src_start)
    copied_rows = 0
    next_report_at = 0.0
    if float(progress_interval_s) > 0:
        next_report_at = time.monotonic() + float(progress_interval_s)

    out_pos = 0
    for s in range(int(src_start), int(src_end), copy_batch_size):
        e = min(int(src_end), s + copy_batch_size)
        data = src_ds[s:e]
        dst_ds[out_pos : out_pos + (e - s)] = data
        out_pos += (e - s)
        copied_rows += (e - s)

        if float(progress_interval_s) > 0:
            now = time.monotonic()
            if now >= next_report_at:
                pct = 100.0 * (float(copied_rows) / float(max(1, total_rows)))
                if progress_context is None:
                    _emit_progress(f"[COPY] {copied_rows}/{total_rows} rows ({pct:.1f}%)")
                else:
                    _emit_progress(f"[COPY] {progress_context}: {copied_rows}/{total_rows} rows ({pct:.1f}%)")
                next_report_at = now + float(progress_interval_s)


def _dataset_name_allowed(dataset_name: str, metadata_mode: str) -> bool:
    if metadata_mode == "full":
        return True
    if metadata_mode != "minimal":
        raise ValueError(f"unsupported metadata_mode: {metadata_mode}")

    keep = {
        "voxels",
        "window_index",
        "window_t_start_us",
        "window_t_end_us",
        "window_rel_start_us",
        "window_rel_end_us",
        "anchor_timestamp_us",
        "anchor_rel_timestamp_us",
        "window_event_count",
        "window_activity_score",
        "window_active_pixel_ratio",
        "window_activity_grid",
        "embedded_segmentation",
        "embedded_semantics",
        "embedded_depth",
        "segmentation_available",
        "segmentation_timestamp_us",
        "segmentation_time_delta_us",
        "segmentation_relpath",
    }
    return dataset_name in keep


def _create_dataset_with_fallback(
    out_h5: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype,
    create_kwargs: dict,
) -> h5py.Dataset:
    try:
        return out_h5.create_dataset(name, shape=shape, dtype=dtype, **create_kwargs)
    except Exception:
        # Some HDF5 dtypes (notably variable-length strings) may not support
        # filter/chunk combinations consistently across environments.
        return out_h5.create_dataset(name, shape=shape, dtype=dtype)


def _should_rebase_window_index(
    *,
    src_h5: h5py.File,
    src_ds: h5py.Dataset,
    src_start: int,
    src_end: int,
) -> bool:
    embedded_name = _load_h5_attr_str(src_h5, "embedded_label_dataset", default="")
    if embedded_name not in {"embedded_semantics", "embedded_depth"}:
        return False
    if embedded_name not in src_h5:
        return False
    values = np.asarray(src_ds[src_start:src_end], dtype=np.int64).reshape(-1)
    if values.size == 0:
        return False
    return bool(np.all((values >= int(src_start)) & (values < int(src_end))))


def _copy_dataset_slice_rebased(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    src_start: int,
    src_end: int,
    copy_batch_size: int,
    offset: int,
    progress_interval_s: float = 0.0,
    progress_context: str | None = None,
) -> None:
    if src_end <= src_start:
        return

    copy_batch_size = max(1, int(copy_batch_size))
    total_rows = int(src_end - src_start)
    copied_rows = 0
    next_report_at = 0.0
    if float(progress_interval_s) > 0:
        next_report_at = time.monotonic() + float(progress_interval_s)

    out_pos = 0
    for s in range(int(src_start), int(src_end), copy_batch_size):
        e = min(int(src_end), s + copy_batch_size)
        data = np.asarray(src_ds[s:e], dtype=np.int64) - int(offset)
        dst_ds[out_pos : out_pos + (e - s)] = data.astype(src_ds.dtype, copy=False)
        out_pos += (e - s)
        copied_rows += (e - s)

        if float(progress_interval_s) > 0:
            now = time.monotonic()
            if now >= next_report_at:
                pct = 100.0 * (float(copied_rows) / float(max(1, total_rows)))
                if progress_context is None:
                    _emit_progress(f"[COPY] {copied_rows}/{total_rows} rows ({pct:.1f}%)")
                else:
                    _emit_progress(f"[COPY] {progress_context}: {copied_rows}/{total_rows} rows ({pct:.1f}%)")
                next_report_at = now + float(progress_interval_s)


def _write_chunk_file(
    *,
    src_h5_path: Path,
    out_h5_path: Path,
    src_start: int,
    src_end: int,
    copy_batch_size: int,
    chunk_index: int,
    total_chunks: int,
    chunk_duration_s: float,
    metadata_mode: str,
    progress_interval_s: float,
    log_chunk_progress: bool,
    log_dataset_progress: bool,
) -> None:
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_output_path(out_h5_path, ".tmp")
    cleanup_tmp_file(tmp_path, context=f"start chunk write {out_h5_path}", strict=True)

    chunk_size = int(src_end - src_start)
    chunk_started_at = time.monotonic()
    pid = os.getpid()
    if bool(log_chunk_progress):
        _emit_progress(
            "[CHUNK START] "
            f"pid={pid} "
            f"{src_h5_path.name} chunk={chunk_index + 1}/{total_chunks} "
            f"rows={chunk_size} out={out_h5_path.name}"
        )

    with h5py.File(str(src_h5_path), "r") as src_h5:
        with h5py.File(str(tmp_path), "w") as out_h5:
            if "voxels" not in src_h5:
                raise KeyError(f"missing voxels in source: {src_h5_path}")
            n_samples = int(src_h5["voxels"].shape[0])
            if src_end > n_samples:
                raise ValueError(
                    f"slice exceeds source size: [{src_start}:{src_end}) vs n_samples={n_samples}, file={src_h5_path}"
                )
            _copy_attrs(src_h5, out_h5)

            group_names: list[str] = []
            dataset_names: list[str] = []

            def _visitor(name: str, obj) -> None:
                if isinstance(obj, h5py.Group):
                    group_names.append(name)
                elif isinstance(obj, h5py.Dataset):
                    dataset_names.append(name)

            src_h5.visititems(_visitor)

            allowed_dataset_names = sorted([n for n in dataset_names if _dataset_name_allowed(n, metadata_mode)])
            required_groups = set()
            for dname in allowed_dataset_names:
                if "/" in dname:
                    parent = dname.rsplit("/", 1)[0]
                    while True:
                        required_groups.add(parent)
                        if "/" not in parent:
                            break
                        parent = parent.rsplit("/", 1)[0]

            for gname in sorted(group_names):
                if gname not in required_groups:
                    continue
                out_group = out_h5.require_group(gname)
                _copy_attrs(src_h5[gname], out_group)

            for dname in allowed_dataset_names:
                src_ds = src_h5[dname]
                row_aligned = src_ds.ndim > 0 and src_ds.shape[0] == n_samples
                if bool(log_dataset_progress):
                    _emit_progress(
                        "[DATASET START] "
                        f"pid={pid} {src_h5_path.name} chunk={chunk_index + 1}/{total_chunks} "
                        f"name={dname} dtype={src_ds.dtype} row_aligned={int(row_aligned)}"
                    )

                if row_aligned:
                    out_shape = (int(src_end - src_start),) + tuple(src_ds.shape[1:])
                else:
                    out_shape = tuple(src_ds.shape)

                parent_name = dname.rsplit("/", 1)[0] if "/" in dname else ""
                if parent_name:
                    out_h5.require_group(parent_name)

                create_kwargs = _dataset_create_kwargs(src_ds, out_shape=out_shape)
                out_ds = _create_dataset_with_fallback(
                    out_h5=out_h5,
                    name=dname,
                    shape=out_shape,
                    dtype=src_ds.dtype,
                    create_kwargs=create_kwargs,
                )
                _copy_attrs(src_ds, out_ds)

                if src_ds.ndim == 0:
                    out_ds[()] = src_ds[()]
                elif row_aligned:
                    ds_progress_interval_s = float(progress_interval_s) if dname == "voxels" else 0.0
                    if dname == "window_index" and _should_rebase_window_index(
                        src_h5=src_h5,
                        src_ds=src_ds,
                        src_start=src_start,
                        src_end=src_end,
                    ):
                        _copy_dataset_slice_rebased(
                            src_ds=src_ds,
                            dst_ds=out_ds,
                            src_start=src_start,
                            src_end=src_end,
                            copy_batch_size=copy_batch_size,
                            offset=src_start,
                            progress_interval_s=ds_progress_interval_s,
                            progress_context=(
                                f"pid={pid} {src_h5_path.name} "
                                f"chunk={chunk_index + 1}/{total_chunks} ds={dname}"
                            ),
                        )
                    else:
                        _copy_dataset_slice(
                            src_ds=src_ds,
                            dst_ds=out_ds,
                            src_start=src_start,
                            src_end=src_end,
                            copy_batch_size=copy_batch_size,
                            progress_interval_s=ds_progress_interval_s,
                            progress_context=(
                                f"pid={pid} {src_h5_path.name} "
                                f"chunk={chunk_index + 1}/{total_chunks} ds={dname}"
                            ),
                        )
                else:
                    out_ds[...] = src_ds[...]
                if bool(log_dataset_progress):
                    _emit_progress(
                        "[DATASET DONE] "
                        f"pid={pid} {src_h5_path.name} chunk={chunk_index + 1}/{total_chunks} name={dname}"
                    )

            out_h5.attrs["split_parent_file"] = str(src_h5_path)
            out_h5.attrs["split_chunk_index"] = int(chunk_index)
            out_h5.attrs["split_num_chunks"] = int(total_chunks)
            out_h5.attrs["split_chunk_duration_s"] = float(chunk_duration_s)
            out_h5.attrs["split_source_start_index"] = int(src_start)
            out_h5.attrs["split_source_end_index_exclusive"] = int(src_end)
            out_h5.attrs["split_source_num_samples"] = int(n_samples)

    os.replace(tmp_path, out_h5_path)
    if bool(log_chunk_progress):
        elapsed = time.monotonic() - chunk_started_at
        _emit_progress(
            "[CHUNK DONE] "
            f"pid={pid} "
            f"{src_h5_path.name} chunk={chunk_index + 1}/{total_chunks} "
            f"rows={chunk_size} elapsed={elapsed:.1f}s"
        )


def _process_one_file(
    *,
    input_path: Path,
    output_base_path: Path,
    chunk_duration_s: float,
    copy_batch_size: int,
    min_windows_per_chunk: int,
    chunk_index_pad: int,
    overwrite: bool,
    metadata_mode: str,
    progress_interval_s: float,
    log_chunk_progress: bool,
    log_dataset_progress: bool,
    delete_source_companion_mp4: bool,
) -> tuple[str, int, int]:
    with h5py.File(str(input_path), "r") as h5f:
        n_samples = int(h5f["voxels"].shape[0])
        anchors = _anchor_times_us(h5f=h5f, n_samples=n_samples)
        companion_mp4_info = _resolve_companion_mp4_info(h5f, input_path)

    chunk_duration_us = int(round(float(chunk_duration_s) * 1e6))
    ranges = _contiguous_chunk_ranges(anchors_us=anchors, chunk_duration_us=chunk_duration_us)
    if int(min_windows_per_chunk) > 1:
        ranges = [(s, e) for (s, e) in ranges if (e - s) >= int(min_windows_per_chunk)]

    done = 0
    companion_fps: float | None = None
    companion_fps_source = ""
    if companion_mp4_info is not None:
        companion_fps, companion_fps_source = _resolved_companion_mp4_fps(companion_mp4_info)

    for chunk_idx, (src_start, src_end) in enumerate(ranges):
        out_name = f"{output_base_path.stem}_part{chunk_idx:0{int(chunk_index_pad)}d}{output_base_path.suffix}"
        out_path = output_base_path.with_name(out_name)
        out_mp4_path = out_path.with_suffix(".mp4")

        need_h5 = bool(overwrite) or not out_path.exists()
        need_mp4 = False
        need_attr_update = False
        if companion_mp4_info is not None:
            need_mp4 = bool(overwrite) or not out_mp4_path.exists()
            need_attr_update = (not need_mp4) and (not _chunk_h5_has_matching_companion(out_path, out_mp4_path))

        if not need_h5 and not need_mp4 and not need_attr_update:
            continue

        if bool(overwrite):
            out_path.unlink(missing_ok=True)
            out_mp4_path.unlink(missing_ok=True)

        try:
            if need_h5:
                _write_chunk_file(
                    src_h5_path=input_path,
                    out_h5_path=out_path,
                    src_start=src_start,
                    src_end=src_end,
                    copy_batch_size=copy_batch_size,
                    chunk_index=chunk_idx,
                    total_chunks=len(ranges),
                    chunk_duration_s=float(chunk_duration_s),
                    metadata_mode=metadata_mode,
                    progress_interval_s=float(progress_interval_s),
                    log_chunk_progress=bool(log_chunk_progress),
                    log_dataset_progress=bool(log_dataset_progress),
                )
                done += 1
            if companion_mp4_info is not None:
                if need_mp4:
                    assert companion_fps is not None
                    _write_chunk_companion_mp4(
                        source_mp4_path=companion_mp4_info.source_path,
                        out_mp4_path=out_mp4_path,
                        src_start=src_start,
                        src_end=src_end,
                        fps=float(companion_fps),
                    )
                if need_h5 or need_mp4 or need_attr_update:
                    assert companion_fps is not None
                    _update_chunk_companion_mp4_attrs(
                        out_h5_path=out_path,
                        out_mp4_path=out_mp4_path,
                        fps=float(companion_fps),
                        fps_source=str(companion_fps_source),
                    )
        except Exception:
            if need_h5:
                out_path.unlink(missing_ok=True)
            if need_mp4:
                out_mp4_path.unlink(missing_ok=True)
            raise

    if bool(delete_source_companion_mp4) and companion_mp4_info is not None:
        companion_mp4_info.source_path.unlink(missing_ok=True)

    return str(input_path), int(n_samples), int(done)


def split_voxel_h5_file(
    *,
    input_path: Path,
    output_base_path: Path,
    chunk_duration_s: float = 20.0,
    copy_batch_size: int = 8,
    min_windows_per_chunk: int = 1,
    chunk_index_pad: int = 4,
    overwrite: bool = False,
    metadata_mode: str = "full",
    progress_interval_s: float = 0.0,
    log_chunk_progress: bool = False,
    log_dataset_progress: bool = False,
    delete_source_companion_mp4: bool = False,
) -> tuple[int, int]:
    _, n_samples, n_written = _process_one_file(
        input_path=input_path,
        output_base_path=output_base_path,
        chunk_duration_s=float(chunk_duration_s),
        copy_batch_size=int(copy_batch_size),
        min_windows_per_chunk=int(min_windows_per_chunk),
        chunk_index_pad=int(chunk_index_pad),
        overwrite=bool(overwrite),
        metadata_mode=str(metadata_mode),
        progress_interval_s=float(progress_interval_s),
        log_chunk_progress=bool(log_chunk_progress),
        log_dataset_progress=bool(log_dataset_progress),
        delete_source_companion_mp4=bool(delete_source_companion_mp4),
    )
    return int(n_samples), int(n_written)


def _worker(job: dict) -> tuple[str, bool, str | None, int, int]:
    try:
        input_path = Path(job["input_path"])
        output_base_path = Path(job["output_base_path"])
        file_path, n_samples, n_written = _process_one_file(
            input_path=input_path,
            output_base_path=output_base_path,
            chunk_duration_s=float(job["chunk_duration_s"]),
            copy_batch_size=int(job["copy_batch_size"]),
            min_windows_per_chunk=int(job["min_windows_per_chunk"]),
            chunk_index_pad=int(job["chunk_index_pad"]),
            overwrite=bool(job["overwrite"]),
            metadata_mode=str(job["metadata_mode"]),
            progress_interval_s=float(job["progress_interval_s"]),
            log_chunk_progress=bool(job["log_chunk_progress"]),
            log_dataset_progress=bool(job["log_dataset_progress"]),
            delete_source_companion_mp4=bool(job.get("delete_source_companion_mp4", False)),
        )
        return file_path, True, None, n_samples, n_written
    except Exception as exc:
        return str(job.get("input_path", "")), False, str(exc), 0, 0


def main() -> None:
    parser = argparse.ArgumentParser("Split preprocessed voxel H5 files into shorter files by duration.")
    parser.add_argument("--input_path", type=Path, default=None, help="Single input voxel H5.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory for batch split.")
    parser.add_argument("--output_root", type=Path, default=None, help="Optional output root (recommended).")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursive scan.")
    parser.add_argument("--file_pattern", type=str, default="*.h5", help="Glob pattern under dataset_root.")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on files to process.")
    parser.add_argument(
        "--chunk_duration_s",
        type=float,
        default=20.0,
        help="Target chunk duration in seconds.",
    )
    parser.add_argument(
        "--min_windows_per_chunk",
        type=int,
        default=1,
        help="Drop chunks with fewer than this number of windows.",
    )
    parser.add_argument("--chunk_index_pad", type=int, default=4, help="Zero padding for part index.")
    parser.add_argument(
        "--copy_batch_size",
        type=int,
        default=8,
        help="Rows copied per I/O batch while slicing datasets (smaller uses less RAM).",
    )
    parser.add_argument(
        "--metadata_mode",
        choices=["full", "minimal"],
        default="full",
        help="full: copy all datasets. minimal: copy voxels and essential timing datasets only.",
    )
    parser.add_argument(
        "--progress_interval_s",
        type=float,
        default=0.0,
        help="If >0, print row-copy progress every N seconds during dataset slicing.",
    )
    parser.add_argument(
        "--log_chunk_progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print chunk start/end logs (useful to see activity when each chunk is slow).",
    )
    parser.add_argument(
        "--log_dataset_progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print dataset-level start/end logs inside each chunk (useful for crash localization).",
    )
    parser.add_argument(
        "--delete_source_companion_mp4",
        action="store_true",
        help="After successful split, also delete the source companion MP4 when one is declared in the H5.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--num_processes", type=int, default=1, help="Parallel workers.")
    args = parser.parse_args()

    if float(args.chunk_duration_s) <= 0:
        raise ValueError("--chunk_duration_s must be > 0")
    if int(args.num_processes) < 1:
        raise ValueError("--num_processes must be >= 1")
    if int(args.min_windows_per_chunk) < 1:
        raise ValueError("--min_windows_per_chunk must be >= 1")
    if int(args.copy_batch_size) < 1:
        raise ValueError("--copy_batch_size must be >= 1")
    if float(args.progress_interval_s) < 0:
        raise ValueError("--progress_interval_s must be >= 0")

    input_files = _select_input_files(
        input_path=args.input_path,
        dataset_root=args.dataset_root,
        recursive=bool(args.recursive),
        file_pattern=str(args.file_pattern),
        max_files=args.max_files,
    )
    if len(input_files) == 0:
        raise FileNotFoundError("no voxel h5 files found to split")

    jobs: list[dict] = []
    for input_path in input_files:
        output_base_path = _resolve_output_path(
            input_path=input_path,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
        output_base_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "input_path": str(input_path),
                "output_base_path": str(output_base_path),
                "chunk_duration_s": float(args.chunk_duration_s),
                "copy_batch_size": int(args.copy_batch_size),
                "min_windows_per_chunk": int(args.min_windows_per_chunk),
                "chunk_index_pad": int(args.chunk_index_pad),
                "overwrite": bool(args.overwrite),
                "metadata_mode": str(args.metadata_mode),
                "progress_interval_s": float(args.progress_interval_s),
                "log_chunk_progress": bool(args.log_chunk_progress),
                "log_dataset_progress": bool(args.log_dataset_progress),
                "delete_source_companion_mp4": bool(args.delete_source_companion_mp4),
            }
        )

    num_ok = 0
    num_fail = 0
    total_input_samples = 0
    total_output_files = 0
    progress_queue = None
    progress_thread: threading.Thread | None = None

    if int(args.num_processes) == 1:
        iterator = (_worker(job) for job in jobs)
        pbar = tqdm.tqdm(iterator, total=len(jobs), desc="split voxel h5")
    else:
        ctx = mp.get_context("spawn")
        progress_enabled = bool(
            args.log_chunk_progress or args.log_dataset_progress or float(args.progress_interval_s) > 0
        )
        if progress_enabled:
            progress_queue = ctx.Queue()
            progress_thread = threading.Thread(
                target=_progress_consumer_loop,
                args=(progress_queue,),
                daemon=True,
            )
            progress_thread.start()

        pool = ctx.Pool(
            processes=int(args.num_processes),
            initializer=_init_worker_progress_queue,
            initargs=(progress_queue,),
        )
        pbar = tqdm.tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="split voxel h5")

    try:
        for file_path, ok, err, n_samples, n_written in pbar:
            if ok:
                num_ok += 1
                total_input_samples += int(n_samples)
                total_output_files += int(n_written)
            else:
                num_fail += 1
                print(f"[FAILED] {file_path}: {err}")
    finally:
        if int(args.num_processes) > 1:
            pool.close()
            pool.join()
        if progress_queue is not None:
            progress_queue.put(None)
        if progress_thread is not None:
            progress_thread.join()

    print(
        "[SUMMARY] "
        f"files_ok={num_ok}, files_failed={num_fail}, "
        f"input_samples={total_input_samples}, output_files={total_output_files}"
    )
    if num_fail > 0:
        raise RuntimeError(f"{num_fail} files failed during split")


if __name__ == "__main__":
    main()
