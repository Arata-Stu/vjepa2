from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import weakref
from pathlib import Path

# Keep CPU math libraries single-threaded per worker process.
# This avoids heavy oversubscription when preprocessing with multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import numpy as np
import torch
import tqdm
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representations import EventVoxelGrid, accumulate_events_to_rgb
from scripts.preprocess.utils import (
    cleanup_tmp_file,
    ensure_scale_tag_in_filename,
    get_h5_compression_flags,
    LazyRgbMp4Writer,
    normalize_polarity_to_binary,
    normalized_output_subdir,
    normalized_output_suffix,
    tmp_media_output_path,
    tmp_output_path,
)
from scripts.preprocess.split_voxel_h5_by_duration import split_voxel_h5_file


H5_COMPRESSION_FLAGS = get_h5_compression_flags()
MS_TO_IDX_BUILD_CHUNK_EVENTS = 5_000_000
ACTIVITY_MODES = {"full", "light"}
REPRESENTATION_MODES = {"voxel_grid", "event_image"}


def _empty_events() -> dict[str, np.ndarray]:
    return {
        "x": np.empty((0,), dtype=np.uint16),
        "y": np.empty((0,), dtype=np.uint16),
        "p": np.empty((0,), dtype=np.uint8),
        "t": np.empty((0,), dtype=np.int64),
    }


def _read_t_offset(filehandle: h5py.File) -> int:
    if "t_offset" not in filehandle:
        return 0
    return int(filehandle["t_offset"][()])


def _extract_from_h5_by_index(filehandle: h5py.File, ev_start_idx: int, ev_end_idx: int) -> dict[str, np.ndarray]:
    events = filehandle["events"]
    t_offset = _read_t_offset(filehandle)

    x = events["x"][ev_start_idx:ev_end_idx]
    y = events["y"][ev_start_idx:ev_end_idx]
    p = events["p"][ev_start_idx:ev_end_idx]
    t = events["t"][ev_start_idx:ev_end_idx].astype(np.int64) + t_offset

    return {
        "x": x,
        "y": y,
        "p": p,
        "t": t,
    }


def _get_time_bounds_us(filehandle: h5py.File) -> tuple[int | None, int | None]:
    t = filehandle["events/t"]
    num_events = len(t)
    if num_events == 0:
        return None, None

    t_offset = _read_t_offset(filehandle)
    t_first = int(t[0]) + t_offset
    t_last_exclusive = int(t[num_events - 1]) + t_offset + 1
    return t_first, t_last_exclusive


def _load_ms_to_idx(filehandle: h5py.File) -> np.ndarray | None:
    if "ms_to_idx" not in filehandle:
        return None
    return filehandle["ms_to_idx"][()]


def _coarse_bounds_from_ms_to_idx(
    ms_to_idx: np.ndarray | None,
    num_events: int,
    t_offset_us: int,
    start_us: int,
    end_us: int,
) -> tuple[int, int]:
    if ms_to_idx is None or ms_to_idx.size == 0:
        return 0, num_events

    # ms_to_idx is indexed by relative event time (events/t), not absolute wall-clock time.
    start_us_rel = int(start_us) - int(t_offset_us)
    end_us_rel = int(end_us) - int(t_offset_us)
    if end_us_rel <= 0:
        return 0, 0
    start_ms = max(int(start_us_rel // 1000), 0)
    end_ms_exclusive = max(int((end_us_rel + 999) // 1000), start_ms + 1)

    start_ms = min(start_ms, ms_to_idx.size - 1)
    start_idx = int(ms_to_idx[start_ms])

    if end_ms_exclusive >= ms_to_idx.size:
        end_idx = num_events
    else:
        end_idx = int(ms_to_idx[end_ms_exclusive])

    start_idx = max(0, min(start_idx, num_events))
    end_idx = max(0, min(end_idx, num_events))
    return start_idx, end_idx


def _build_ms_to_idx_from_events_t(
    filehandle: h5py.File,
    chunk_events: int = MS_TO_IDX_BUILD_CHUNK_EVENTS,
) -> np.ndarray:
    t_ds = filehandle["events/t"]
    num_events = len(t_ds)
    if num_events == 0:
        return np.zeros((0,), dtype="uint64")

    max_t_us = int(t_ds[num_events - 1])
    max_ms = int(max_t_us // 1000)
    counts_per_ms = np.zeros((max_ms + 1,), dtype=np.uint64)
    step = max(1, int(chunk_events))

    for start in range(0, num_events, step):
        end = min(start + step, num_events)
        t_chunk = np.asarray(t_ds[start:end], dtype=np.int64)
        if t_chunk.size == 0:
            continue
        ms_chunk = (t_chunk // 1000).astype(np.int64, copy=False)
        unique_ms, unique_counts = np.unique(ms_chunk, return_counts=True)
        counts_per_ms[unique_ms] += unique_counts.astype(np.uint64, copy=False)

    ms_to_idx = np.zeros((max_ms + 2,), dtype=np.uint64)
    ms_to_idx[1:] = counts_per_ms
    ms_to_idx = ms_to_idx[:-1].cumsum()
    return ms_to_idx


def _extract_events_by_time(
    filehandle: h5py.File,
    start_us: int,
    end_us: int,
    ms_to_idx: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if end_us <= start_us:
        return _empty_events()

    t_ds = filehandle["events/t"]
    num_events = len(t_ds)
    if num_events == 0:
        return _empty_events()

    t_offset = _read_t_offset(filehandle)
    coarse_start, coarse_end = _coarse_bounds_from_ms_to_idx(
        ms_to_idx=ms_to_idx,
        num_events=num_events,
        t_offset_us=t_offset,
        start_us=start_us,
        end_us=end_us,
    )
    if coarse_end <= coarse_start:
        return _empty_events()

    t_coarse_abs = t_ds[coarse_start:coarse_end].astype(np.int64) + t_offset
    if t_coarse_abs.size == 0:
        return _empty_events()

    rel_start = int(np.searchsorted(t_coarse_abs, start_us, side="left"))
    rel_end = int(np.searchsorted(t_coarse_abs, end_us, side="left"))
    ev_start_idx = coarse_start + rel_start
    ev_end_idx = coarse_start + rel_end

    if ev_end_idx <= ev_start_idx:
        return _empty_events()
    return _extract_from_h5_by_index(filehandle, ev_start_idx=ev_start_idx, ev_end_idx=ev_end_idx)


def _events_to_voxel_numpy(
    events: dict[str, np.ndarray],
    voxelizer: EventVoxelGrid,
    input_height: int,
    input_width: int,
    downsample_factor: int,
) -> np.ndarray:
    processed = _downsample_events_nearest(
        events=events,
        input_height=input_height,
        input_width=input_width,
        downsample_factor=downsample_factor,
    )
    if len(processed["t"]) == 0:
        return np.zeros(voxelizer.voxel_grid.shape, dtype=np.float32)

    # Convert to float tensors once to match interpolation arithmetic inside voxelizer.
    # Shift time to improve float precision.
    t_shifted = (processed["t"] - processed["t"][0]).astype(np.float32, copy=False)
    events_torch = {
        "x": torch.from_numpy(processed["x"].astype(np.float32, copy=False)),
        "y": torch.from_numpy(processed["y"].astype(np.float32, copy=False)),
        "p": torch.from_numpy(processed["p"].astype(np.float32, copy=False)),
        "t": torch.from_numpy(t_shifted),
    }
    voxel = voxelizer.convert(events_torch)
    return voxel.cpu().numpy().astype(np.float32, copy=False)


def _events_to_event_image_numpy(
    events: dict[str, np.ndarray],
    input_height: int,
    input_width: int,
    downsample_factor: int,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    output_height, output_width = _resolve_output_resolution(
        input_height=input_height,
        input_width=input_width,
        downsample_factor=downsample_factor,
    )
    normalized = _downsample_events_nearest(
        events=events,
        input_height=input_height,
        input_width=input_width,
        downsample_factor=downsample_factor,
    )
    return accumulate_events_to_rgb(
        normalized["x"],
        normalized["y"],
        normalized["p"],
        (output_height, output_width),
        percentile=percentile,
        dtype=np.float32,
    )


def _resolve_output_resolution(input_height: int, input_width: int, downsample_factor: int) -> tuple[int, int]:
    if int(downsample_factor) < 1:
        raise ValueError("downsample_factor must be >= 1")
    if int(downsample_factor) == 1:
        return int(input_height), int(input_width)

    if input_height % int(downsample_factor) != 0 or input_width % int(downsample_factor) != 0:
        raise ValueError(
            "input resolution must be divisible by downsample_factor for nearest downsample: "
            f"{input_width}x{input_height}, factor={downsample_factor}"
        )
    return int(input_height // int(downsample_factor)), int(input_width // int(downsample_factor))


def _downsample_events_nearest(
    events: dict[str, np.ndarray],
    input_height: int,
    input_width: int,
    downsample_factor: int,
) -> dict[str, np.ndarray]:
    if len(events["t"]) == 0:
        return _empty_events()

    out_height, out_width = _resolve_output_resolution(
        input_height=input_height,
        input_width=input_width,
        downsample_factor=downsample_factor,
    )
    x_src = events["x"].astype(np.int64, copy=False)
    y_src = events["y"].astype(np.int64, copy=False)
    p_src = events["p"]
    t_src = events["t"]

    valid_in = (x_src >= 0) & (x_src < int(input_width)) & (y_src >= 0) & (y_src < int(input_height))
    if not np.any(valid_in):
        return _empty_events()

    x_src = x_src[valid_in]
    y_src = y_src[valid_in]
    p_src = p_src[valid_in]
    t_src = t_src[valid_in]

    if int(downsample_factor) == 1:
        x_out = x_src.astype(np.float32, copy=False)
        y_out = y_src.astype(np.float32, copy=False)
    else:
        # Nearest-neighbor downsample by integer factor (e.g., factor=2 for 1/2 resolution).
        x_out = (x_src // int(downsample_factor)).astype(np.float32, copy=False)
        y_out = (y_src // int(downsample_factor)).astype(np.float32, copy=False)

    valid_out = (
        (x_out >= 0)
        & (x_out < float(out_width))
        & (y_out >= 0)
        & (y_out < float(out_height))
    )
    if not np.any(valid_out):
        return _empty_events()

    p_bin = normalize_polarity_to_binary(p_src[valid_out], dtype=np.uint8)
    return {
        "x": x_out[valid_out],
        "y": y_out[valid_out],
        "p": p_bin,
        "t": t_src[valid_out].astype(np.int64, copy=False),
    }


def _load_image_timestamps(image_timestamps_path: Path) -> np.ndarray:
    if not image_timestamps_path.exists():
        raise FileNotFoundError(f"image timestamps file not found: {image_timestamps_path}")

    timestamps = np.loadtxt(str(image_timestamps_path), dtype=np.int64)
    timestamps = np.atleast_1d(timestamps).astype(np.int64, copy=False).reshape(-1)
    return timestamps


def _build_fixed_windows(
    t_first_us: int,
    t_last_exclusive_us: int,
    accum_time_us: int,
    stride_time_us: int,
) -> list[tuple[int, int, int]]:
    starts = np.arange(t_first_us, t_last_exclusive_us, stride_time_us, dtype=np.int64)
    windows: list[tuple[int, int, int]] = []
    for start_us in starts:
        start_int = int(start_us)
        end_int = min(start_int + int(accum_time_us), int(t_last_exclusive_us))
        anchor_int = start_int + (end_int - start_int) // 2
        windows.append((start_int, end_int, anchor_int))
    return windows


def _build_image_middle_windows(
    t_first_us: int,
    t_last_exclusive_us: int,
    image_timestamps_us: np.ndarray,
) -> list[tuple[int, int, int]]:
    if image_timestamps_us.size == 0:
        return []

    image_timestamps_us = image_timestamps_us.astype(np.int64, copy=False)
    midpoints = np.empty((max(image_timestamps_us.size - 1, 0),), dtype=np.int64)
    for i in range(image_timestamps_us.size - 1):
        # Use midpoint between adjacent image timestamps as temporal boundary.
        midpoints[i] = image_timestamps_us[i] + (image_timestamps_us[i + 1] - image_timestamps_us[i]) // 2

    windows: list[tuple[int, int, int]] = []
    for i in range(image_timestamps_us.size):
        if i == 0:
            start_us = int(t_first_us)
        else:
            start_us = int(midpoints[i - 1])

        if i < image_timestamps_us.size - 1:
            end_us = int(midpoints[i])
        else:
            end_us = int(t_last_exclusive_us)

        start_us = max(start_us, int(t_first_us))
        end_us = min(end_us, int(t_last_exclusive_us))
        anchor_us = int(image_timestamps_us[i])
        windows.append((start_us, end_us, anchor_us))
    return windows


def _resolve_segmentation_timestamps_path(segmentation_dir: Path) -> Path | None:
    candidates: list[Path] = [
        segmentation_dir / "timestamps.txt",
        segmentation_dir / "semantic_timestamps.txt",
        segmentation_dir.parent / "timestamps.txt",
        segmentation_dir.parent / "semantic_timestamps.txt",
    ]
    if segmentation_dir.parent.name == "semantic":
        sequence_name = segmentation_dir.parent.parent.name
        candidates.append(segmentation_dir.parent / f"{sequence_name}_semantic_timestamps.txt")
    candidates.extend(sorted(segmentation_dir.parent.glob("*_semantic_timestamps.txt")))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _load_segmentation_index(segmentation_dir: Path) -> tuple[np.ndarray, list[str]]:
    if not segmentation_dir.exists() or not segmentation_dir.is_dir():
        return np.empty((0,), dtype=np.int64), []

    label_paths = sorted(segmentation_dir.glob("*.png"))
    if len(label_paths) == 0:
        return np.empty((0,), dtype=np.int64), []

    timestamps_path = _resolve_segmentation_timestamps_path(segmentation_dir)
    if timestamps_path is not None:
        timestamps = _load_image_timestamps(timestamps_path)
        if int(timestamps.size) != len(label_paths):
            raise ValueError(
                "segmentation timestamp count mismatch: "
                f"{timestamps_path} has {int(timestamps.size)} entries, "
                f"but {segmentation_dir} has {len(label_paths)} PNGs"
            )
        return timestamps.astype(np.int64, copy=False), [label_path.name for label_path in label_paths]

    timestamped_files: list[tuple[int, str]] = []
    for label_path in label_paths:
        try:
            timestamped_files.append((int(label_path.stem), label_path.name))
        except ValueError:
            continue

    if len(timestamped_files) == 0:
        return np.empty((0,), dtype=np.int64), []

    timestamped_files.sort(key=lambda x: x[0])
    timestamps = np.array([t for t, _ in timestamped_files], dtype=np.int64)
    relpaths = [n for _, n in timestamped_files]
    return timestamps, relpaths


def _match_segmentation_timestamp(
    anchor_us: int,
    seg_timestamps_us: np.ndarray,
    seg_relpaths: list[str],
    tolerance_us: int,
) -> tuple[int, int, int, str]:
    if seg_timestamps_us.size == 0:
        return 0, -1, -1, ""

    idx = int(np.searchsorted(seg_timestamps_us, anchor_us, side="left"))
    candidate_indices: list[int] = []
    if idx < seg_timestamps_us.size:
        candidate_indices.append(idx)
    if idx > 0:
        candidate_indices.append(idx - 1)

    best_idx = min(candidate_indices, key=lambda k: abs(int(seg_timestamps_us[k]) - int(anchor_us)))
    matched_ts = int(seg_timestamps_us[best_idx])
    delta_us = int(anchor_us - matched_ts)
    available = int(abs(delta_us) <= int(tolerance_us))
    relpath = seg_relpaths[best_idx] if available else ""
    return available, matched_ts, delta_us, relpath


def _load_segmentation_label(segmentation_dir: Path, relpath: str) -> np.ndarray:
    label_path = segmentation_dir / relpath
    with Image.open(str(label_path)) as img:
        label = np.asarray(img)
    if label.ndim == 2:
        return label
    if label.ndim == 3 and label.shape[0] == 1:
        return label[0]
    if label.ndim == 3 and label.shape[-1] == 1:
        return label[..., 0]
    raise ValueError(
        f"Unsupported DSEC segmentation shape={label.shape} at {label_path}. "
        "Expected class-index map (H,W) or singleton-channel variant."
    )


def _event_image_chw_to_hwc_uint8(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected event image [3,H,W], got shape={image_chw.shape}")
    return np.moveaxis(np.asarray(image_chw), 0, -1)


def _compute_activity_metadata(
    voxel: np.ndarray,
    *,
    temporal_bins: int,
    split_polarity: bool,
    spatial_patch_size: int,
    temporal_patch_size: int,
    activity_mode: str,
) -> tuple[np.ndarray, float, float]:
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")
    channels, height, width = voxel.shape
    if split_polarity:
        activity_volume = np.abs(voxel).reshape(2, temporal_bins, height, width).sum(axis=0)
    else:
        activity_volume = np.abs(voxel)
    return _compute_activity_metadata_from_volume(
        activity_volume=activity_volume,
        spatial_patch_size=spatial_patch_size,
        temporal_patch_size=temporal_patch_size,
        activity_mode=activity_mode,
    )


def _compute_activity_metadata_from_volume(
    activity_volume: np.ndarray,
    *,
    spatial_patch_size: int,
    temporal_patch_size: int,
    activity_mode: str,
) -> tuple[np.ndarray, float, float]:
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")
    if activity_volume.ndim != 3:
        raise ValueError(f"activity_volume must be [T,H,W], got shape={activity_volume.shape}")

    temporal_bins, height, width = activity_volume.shape
    nonzero_voxel_ratio = float(np.count_nonzero(activity_volume) / float(max(1, activity_volume.size)))
    active_pixel_ratio = float(np.count_nonzero(activity_volume.sum(axis=0) > 0) / float(max(1, height * width)))
    if activity_mode == "light":
        spatial_volume = activity_volume.sum(axis=0, dtype=np.float32)
        hp = (height + spatial_patch_size - 1) // spatial_patch_size
        wp = (width + spatial_patch_size - 1) // spatial_patch_size
        padded = np.pad(
            spatial_volume,
            ((0, hp * spatial_patch_size - height), (0, wp * spatial_patch_size - width)),
            mode="constant",
        )
        grid = padded.reshape(hp, spatial_patch_size, wp, spatial_patch_size).sum(axis=(1, 3))
        return grid.astype(np.float16, copy=False), nonzero_voxel_ratio, active_pixel_ratio
    tp = (temporal_bins + temporal_patch_size - 1) // temporal_patch_size
    hp = (height + spatial_patch_size - 1) // spatial_patch_size
    wp = (width + spatial_patch_size - 1) // spatial_patch_size
    padded = np.pad(
        activity_volume,
        (
            (0, tp * temporal_patch_size - temporal_bins),
            (0, hp * spatial_patch_size - height),
            (0, wp * spatial_patch_size - width),
        ),
        mode="constant",
    )
    grid = padded.reshape(tp, temporal_patch_size, hp, spatial_patch_size, wp, spatial_patch_size).sum(axis=(1, 3, 5))
    return grid.astype(np.float16, copy=False), nonzero_voxel_ratio, active_pixel_ratio


class VoxelH5Writer:
    def __init__(
        self,
        outfile: Path,
        t_bins: int,
        height: int,
        width: int,
        voxel_dtype: np.dtype,
        with_segmentation_meta: bool = False,
        with_embedded_segmentation: bool = False,
        embedded_segmentation_shape: tuple[int, int] = (1, 1),
        embedded_segmentation_dtype: np.dtype = np.uint8,
        activity_grid_shape: tuple[int, ...] = (1, 1, 1),
        initial_capacity: int = 256,
    ):
        if outfile.exists():
            raise FileExistsError(f"output already exists: {outfile}")

        self.h5f = h5py.File(str(outfile), "a")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        self._capacity = int(initial_capacity)
        self._num_windows = 0
        self._with_segmentation_meta = bool(with_segmentation_meta)
        self._with_embedded_segmentation = bool(with_embedded_segmentation)
        self._datasets: list[str] = [
            "voxels",
            "window_t_start_us",
            "window_t_end_us",
            "window_event_count",
            "anchor_timestamp_us",
            "window_activity_score",
            "window_active_pixel_ratio",
            "window_activity_grid",
        ]

        voxel_chunks = (1, t_bins, min(height, 64), min(width, 64))
        scalar_chunks = (min(self._capacity, 4096),)

        self.h5f.create_dataset(
            "voxels",
            shape=(self._capacity, t_bins, height, width),
            maxshape=(None, t_bins, height, width),
            dtype=voxel_dtype,
            chunks=voxel_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_t_start_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_t_end_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_event_count",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "anchor_timestamp_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_activity_score",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="f4",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_active_pixel_ratio",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="f4",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_activity_grid",
            shape=(self._capacity,) + tuple(activity_grid_shape),
            maxshape=(None,) + tuple(activity_grid_shape),
            dtype="f2",
            chunks=(1,) + tuple(activity_grid_shape),
            **H5_COMPRESSION_FLAGS,
        )
        if self._with_embedded_segmentation:
            seg_h, seg_w = embedded_segmentation_shape
            self.h5f.create_dataset(
                "embedded_segmentation",
                shape=(self._capacity, int(seg_h), int(seg_w)),
                maxshape=(None, int(seg_h), int(seg_w)),
                dtype=embedded_segmentation_dtype,
                chunks=(1, min(int(seg_h), 256), min(int(seg_w), 256)),
                **H5_COMPRESSION_FLAGS,
            )
            self._datasets.append("embedded_segmentation")

        if self._with_segmentation_meta:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            self.h5f.create_dataset(
                "segmentation_available",
                shape=(self._capacity,),
                maxshape=(None,),
                dtype="u1",
                chunks=scalar_chunks,
                **H5_COMPRESSION_FLAGS,
            )
            self.h5f.create_dataset(
                "segmentation_timestamp_us",
                shape=(self._capacity,),
                maxshape=(None,),
                dtype="i8",
                chunks=scalar_chunks,
                **H5_COMPRESSION_FLAGS,
            )
            self.h5f.create_dataset(
                "segmentation_time_delta_us",
                shape=(self._capacity,),
                maxshape=(None,),
                dtype="i8",
                chunks=scalar_chunks,
                **H5_COMPRESSION_FLAGS,
            )
            self.h5f.create_dataset(
                "segmentation_relpath",
                shape=(self._capacity,),
                maxshape=(None,),
                dtype=string_dtype,
                chunks=scalar_chunks,
            )
            self._datasets.extend(
                [
                    "segmentation_available",
                    "segmentation_timestamp_us",
                    "segmentation_time_delta_us",
                    "segmentation_relpath",
                ]
            )

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def _ensure_capacity(self, needed: int):
        if needed <= self._capacity:
            return

        new_capacity = self._capacity
        while new_capacity < needed:
            new_capacity *= 2

        for dset_name in self._datasets:
            self.h5f[dset_name].resize(new_capacity, axis=0)
        self._capacity = new_capacity

    def add_window(
        self,
        voxel: np.ndarray,
        t_start_us: int,
        t_end_us: int,
        event_count: int,
        anchor_timestamp_us: int,
        activity_score: float,
        active_pixel_ratio: float,
        activity_grid: np.ndarray,
        embedded_segmentation: np.ndarray | None = None,
        segmentation_available: int = 0,
        segmentation_timestamp_us: int = -1,
        segmentation_time_delta_us: int = -1,
        segmentation_relpath: str = "",
    ):
        idx = self._num_windows
        self._ensure_capacity(idx + 1)

        self.h5f["voxels"][idx] = voxel
        self.h5f["window_t_start_us"][idx] = int(t_start_us)
        self.h5f["window_t_end_us"][idx] = int(t_end_us)
        self.h5f["window_event_count"][idx] = int(event_count)
        self.h5f["anchor_timestamp_us"][idx] = int(anchor_timestamp_us)
        self.h5f["window_activity_score"][idx] = float(activity_score)
        self.h5f["window_active_pixel_ratio"][idx] = float(active_pixel_ratio)
        self.h5f["window_activity_grid"][idx] = activity_grid
        if self._with_embedded_segmentation:
            if embedded_segmentation is None:
                raise ValueError("embedded_segmentation is required when with_embedded_segmentation=True")
            self.h5f["embedded_segmentation"][idx] = embedded_segmentation
        if self._with_segmentation_meta:
            self.h5f["segmentation_available"][idx] = int(segmentation_available)
            self.h5f["segmentation_timestamp_us"][idx] = int(segmentation_timestamp_us)
            self.h5f["segmentation_time_delta_us"][idx] = int(segmentation_time_delta_us)
            self.h5f["segmentation_relpath"][idx] = segmentation_relpath
        self._num_windows += 1

    def _trim(self):
        for dset_name in self._datasets:
            self.h5f[dset_name].resize(self._num_windows, axis=0)

    def close(self):
        if self._finalizer.alive:
            self._trim()
            self._finalizer()


def process_single_file(
    input_path: Path,
    output_path: Path,
    height: int,
    width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    window_mode: str,
    image_timestamps_path: Path | None,
    normalize: bool,
    output_dtype: str,
    use_trilinear: bool,
    representation: str,
    event_image_percentile: float,
    save_mp4: bool,
    mp4_fps: float | None,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
    sync_segmentation: bool,
    segmentation_dir: Path | None,
    segmentation_tolerance_us: int,
    show_pbar: bool = True,
    tmp_suffix: str = ".tmp",
):
    if t_bins <= 0:
        raise ValueError("t_bins must be > 0")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be > 0")
    if int(downsample_factor) not in (1, 2):
        raise ValueError("downsample_factor must be 1 or 2")
    if accum_time <= 0:
        raise ValueError("accum_time must be > 0")
    if stride_time <= 0:
        raise ValueError("stride_time must be > 0")
    if segmentation_tolerance_us < 0:
        raise ValueError("segmentation_tolerance_us must be >= 0")
    if window_mode not in ("fixed", "image_middle"):
        raise ValueError(f"unsupported window_mode: {window_mode}")
    if window_mode == "image_middle" and image_timestamps_path is None:
        raise ValueError("window_mode=image_middle requires image_timestamps_path")
    if representation not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported representation: {representation}")
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")

    output_height, output_width = _resolve_output_resolution(
        input_height=height,
        input_width=width,
        downsample_factor=downsample_factor,
    )
    representation_channels = int(t_bins) * (2 if split_polarity else 1) if representation == "voxel_grid" else 3
    activity_temporal_bins = int(t_bins) if representation == "voxel_grid" else 1
    voxel_dtype = np.float16 if output_dtype == "float16" else np.float32
    activity_grid_shape = (
        (
            (int(activity_temporal_bins) + int(activity_temporal_patch_size) - 1) // int(activity_temporal_patch_size),
            (int(output_height) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
            (int(output_width) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
        )
        if activity_mode == "full"
        else (
            (int(output_height) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
            (int(output_width) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
        )
    )
    if sync_segmentation and (segmentation_dir is None or not segmentation_dir.exists()):
        raise RuntimeError(
            "DSEC preprocessing now requires self-contained embedded segmentation when "
            f"sync_segmentation=true, but segmentation_dir is unavailable for {input_path}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    cleanup_tmp_file(tmp_path=tmp_path, context=f"start processing {input_path}", strict=True)
    mp4_path = output_path.with_suffix(".mp4")
    tmp_mp4_path = tmp_media_output_path(output_path=mp4_path, tmp_suffix=tmp_suffix)
    if save_mp4 and representation != "event_image":
        raise ValueError("save_mp4 is only supported when representation=event_image")
    if save_mp4:
        cleanup_tmp_file(tmp_path=tmp_mp4_path, context=f"start MP4 processing {input_path}", strict=True)

    writer = None
    mp4_writer = None
    pbar = None
    try:
        with h5py.File(str(input_path), "r") as h5f:
            t_first, t_last_exclusive = _get_time_bounds_us(h5f)
            ms_to_idx = _load_ms_to_idx(h5f)
            ms_to_idx_source = "input"
            if ms_to_idx is None:
                ms_to_idx = _build_ms_to_idx_from_events_t(
                    filehandle=h5f,
                    chunk_events=MS_TO_IDX_BUILD_CHUNK_EVENTS,
                )
                ms_to_idx_source = "generated"

            if t_first is None or t_last_exclusive is None:
                writer = VoxelH5Writer(
                    outfile=tmp_path,
                    t_bins=representation_channels,
                    height=output_height,
                    width=output_width,
                    voxel_dtype=voxel_dtype,
                    with_segmentation_meta=sync_segmentation,
                    activity_grid_shape=activity_grid_shape,
                )
                writer.h5f.attrs["representation"] = "event_voxel_grid" if representation == "voxel_grid" else "event_image"
                writer.h5f.attrs["representation_kind"] = str(representation)
                writer.h5f.attrs["input_height"] = int(height)
                writer.h5f.attrs["input_width"] = int(width)
                writer.h5f.attrs["height"] = int(output_height)
                writer.h5f.attrs["width"] = int(output_width)
                writer.h5f.attrs["downsample_factor"] = int(downsample_factor)
                writer.h5f.attrs["spatial_resize_mode"] = "nearest"
                writer.h5f.attrs["t_bins"] = int(activity_temporal_bins)
                writer.h5f.attrs["voxel_channels"] = int(representation_channels)
                writer.h5f.attrs["split_polarity"] = int(split_polarity if representation == "voxel_grid" else False)
                writer.h5f.attrs["polarity_channels"] = 2
                writer.h5f.attrs["window_mode"] = window_mode
                writer.h5f.attrs["accum_time_us"] = int(accum_time)
                writer.h5f.attrs["stride_time_us"] = int(stride_time)
                writer.h5f.attrs["normalize"] = int(normalize)
                writer.h5f.attrs["trilinear_interpolation"] = int(use_trilinear)
                writer.h5f.attrs["event_image_percentile"] = float(event_image_percentile)
                writer.h5f.attrs["sync_segmentation"] = int(sync_segmentation)
                writer.h5f.attrs["segmentation_tolerance_us"] = int(segmentation_tolerance_us)
                writer.h5f.attrs["image_timestamps_path"] = str(image_timestamps_path) if image_timestamps_path is not None else ""
                writer.h5f.attrs["ms_to_idx_source"] = ms_to_idx_source
                writer.h5f.attrs["activity_mode"] = str(activity_mode)
                writer.h5f.attrs["activity_spatial_patch_size"] = int(activity_spatial_patch_size)
                writer.h5f.attrs["activity_temporal_patch_size"] = int(activity_temporal_patch_size)
                writer.h5f.attrs["has_companion_mp4"] = int(save_mp4)
                writer.h5f.attrs["companion_mp4_relpath"] = mp4_path.name if save_mp4 else ""
                writer.h5f.attrs["companion_mp4_fps"] = 0.0
                writer.h5f.attrs["companion_mp4_fps_source"] = ""
                writer.h5f.attrs["time_origin_us"] = -1
                writer.close()
                writer = None
                os.replace(tmp_path, output_path)
                return

            time_origin_us = int(t_first) if start_time_us is None else int(start_time_us)

            if window_mode == "fixed":
                window_start_us = max(int(t_first), time_origin_us)
                windows = _build_fixed_windows(
                    t_first_us=window_start_us,
                    t_last_exclusive_us=t_last_exclusive,
                    accum_time_us=accum_time,
                    stride_time_us=stride_time,
                )
                image_timestamps = None
            else:
                image_timestamps = _load_image_timestamps(image_timestamps_path)
                windows = _build_image_middle_windows(
                    t_first_us=t_first,
                    t_last_exclusive_us=t_last_exclusive,
                    image_timestamps_us=image_timestamps,
                )

            if sync_segmentation and segmentation_dir is not None:
                seg_timestamps, seg_relpaths = _load_segmentation_index(segmentation_dir)
            else:
                seg_timestamps, seg_relpaths = np.empty((0,), dtype=np.int64), []

            embedded_segmentation_shape: tuple[int, int] | None = None
            embedded_segmentation_dtype: np.dtype | None = None
            if sync_segmentation and segmentation_dir is not None:
                for rel in seg_relpaths:
                    if not rel:
                        continue
                    sample_label = _load_segmentation_label(segmentation_dir, rel)
                    embedded_segmentation_shape = tuple(int(v) for v in sample_label.shape)
                    embedded_segmentation_dtype = sample_label.dtype
                    break
            if sync_segmentation and embedded_segmentation_shape is None:
                raise RuntimeError(
                    "DSEC preprocessing now requires self-contained embedded segmentation when "
                    f"sync_segmentation=true, but no valid segmentation PNGs were found in {segmentation_dir}."
                )

            writer = VoxelH5Writer(
                outfile=tmp_path,
                t_bins=representation_channels,
                height=output_height,
                width=output_width,
                voxel_dtype=voxel_dtype,
                with_segmentation_meta=sync_segmentation,
                with_embedded_segmentation=embedded_segmentation_shape is not None,
                embedded_segmentation_shape=embedded_segmentation_shape or (1, 1),
                embedded_segmentation_dtype=(embedded_segmentation_dtype if embedded_segmentation_dtype is not None else np.uint8),
                activity_grid_shape=activity_grid_shape,
            )
            writer.h5f.attrs["representation"] = "event_voxel_grid" if representation == "voxel_grid" else "event_image"
            writer.h5f.attrs["representation_kind"] = str(representation)
            writer.h5f.attrs["input_height"] = int(height)
            writer.h5f.attrs["input_width"] = int(width)
            writer.h5f.attrs["height"] = int(output_height)
            writer.h5f.attrs["width"] = int(output_width)
            writer.h5f.attrs["downsample_factor"] = int(downsample_factor)
            writer.h5f.attrs["spatial_resize_mode"] = "nearest"
            writer.h5f.attrs["t_bins"] = int(activity_temporal_bins)
            writer.h5f.attrs["voxel_channels"] = int(representation_channels)
            writer.h5f.attrs["split_polarity"] = int(split_polarity if representation == "voxel_grid" else False)
            writer.h5f.attrs["polarity_channels"] = 2
            writer.h5f.attrs["window_mode"] = window_mode
            writer.h5f.attrs["accum_time_us"] = int(accum_time)
            writer.h5f.attrs["stride_time_us"] = int(stride_time)
            writer.h5f.attrs["normalize"] = int(normalize)
            writer.h5f.attrs["trilinear_interpolation"] = int(use_trilinear)
            writer.h5f.attrs["event_image_percentile"] = float(event_image_percentile)
            writer.h5f.attrs["sync_segmentation"] = int(sync_segmentation)
            writer.h5f.attrs["segmentation_tolerance_us"] = int(segmentation_tolerance_us)
            writer.h5f.attrs["image_timestamps_path"] = str(image_timestamps_path) if image_timestamps_path is not None else ""
            writer.h5f.attrs["ms_to_idx_source"] = ms_to_idx_source
            writer.h5f.attrs["activity_mode"] = str(activity_mode)
            writer.h5f.attrs["activity_spatial_patch_size"] = int(activity_spatial_patch_size)
            writer.h5f.attrs["activity_temporal_patch_size"] = int(activity_temporal_patch_size)
            writer.h5f.attrs["has_companion_mp4"] = int(save_mp4)
            writer.h5f.attrs["companion_mp4_relpath"] = mp4_path.name if save_mp4 else ""
            writer.h5f.attrs["companion_mp4_fps"] = 0.0
            writer.h5f.attrs["companion_mp4_fps_source"] = ""
            writer.h5f.attrs["time_origin_us"] = int(time_origin_us)
            if image_timestamps is not None:
                writer.h5f.attrs["num_image_timestamps"] = int(len(image_timestamps))
            if embedded_segmentation_shape is not None:
                writer.h5f.attrs["embedded_label_dataset"] = "embedded_segmentation"
                writer.h5f.attrs["embedded_label_source_path"] = ""

            voxelizer = None
            if representation == "voxel_grid":
                voxelizer = EventVoxelGrid(
                    input_size=(t_bins, output_height, output_width),
                    normalize=normalize,
                    separate_polarity=split_polarity,
                    trilinear_interpolation=use_trilinear,
                )

            if show_pbar:
                pbar = tqdm.tqdm(total=len(windows), desc=input_path.name, leave=False)

            for start_us_int, end_us_int, anchor_us_int in windows:
                events = _extract_events_by_time(
                    filehandle=h5f,
                    start_us=start_us_int,
                    end_us=end_us_int,
                    ms_to_idx=ms_to_idx,
                )
                if representation == "voxel_grid":
                    assert voxelizer is not None
                    window_tensor = _events_to_voxel_numpy(
                        events=events,
                        voxelizer=voxelizer,
                        input_height=height,
                        input_width=width,
                        downsample_factor=downsample_factor,
                    )
                    activity_grid, activity_score, active_pixel_ratio = _compute_activity_metadata(
                        voxel=window_tensor,
                        temporal_bins=int(t_bins),
                        split_polarity=bool(split_polarity),
                        spatial_patch_size=int(activity_spatial_patch_size),
                        temporal_patch_size=int(activity_temporal_patch_size),
                        activity_mode=str(activity_mode),
                    )
                else:
                    window_tensor, activity_volume = _events_to_event_image_numpy(
                        events=events,
                        input_height=height,
                        input_width=width,
                        downsample_factor=downsample_factor,
                        percentile=float(event_image_percentile),
                    )
                    activity_grid, activity_score, active_pixel_ratio = _compute_activity_metadata_from_volume(
                        activity_volume=activity_volume,
                        spatial_patch_size=int(activity_spatial_patch_size),
                        temporal_patch_size=int(activity_temporal_patch_size),
                        activity_mode=str(activity_mode),
                    )
                    if save_mp4:
                        frame_rgb = _event_image_chw_to_hwc_uint8(window_tensor)
                        if mp4_writer is None:
                            mp4_writer = LazyRgbMp4Writer(
                                tmp_mp4_path,
                                fps=mp4_fps,
                            )
                        mp4_writer.write_rgb(frame_rgb, timestamp_us=anchor_us_int)
                window_tensor = window_tensor.astype(voxel_dtype, copy=False)
                seg_available = 0
                seg_timestamp = -1
                seg_delta = -1
                seg_relpath = ""
                embedded_segmentation = None
                if sync_segmentation:
                    seg_available, seg_timestamp, seg_delta, seg_relpath = _match_segmentation_timestamp(
                        anchor_us=anchor_us_int,
                        seg_timestamps_us=seg_timestamps,
                        seg_relpaths=seg_relpaths,
                        tolerance_us=segmentation_tolerance_us,
                    )
                    if seg_available and segmentation_dir is not None and embedded_segmentation_shape is not None:
                        embedded_segmentation = _load_segmentation_label(segmentation_dir, seg_relpath)
                        if embedded_segmentation.shape != embedded_segmentation_shape:
                            raise ValueError(
                                "Inconsistent DSEC segmentation shape: "
                                f"expected {embedded_segmentation_shape}, got {embedded_segmentation.shape} "
                                f"for {segmentation_dir / seg_relpath}"
                            )
                    elif embedded_segmentation_shape is not None:
                        embedded_segmentation = np.full(
                            embedded_segmentation_shape,
                            fill_value=255,
                            dtype=embedded_segmentation_dtype if embedded_segmentation_dtype is not None else np.uint8,
                        )
                writer.add_window(
                    voxel=window_tensor,
                    t_start_us=start_us_int,
                    t_end_us=end_us_int,
                    event_count=len(events["t"]),
                    anchor_timestamp_us=anchor_us_int,
                    activity_score=activity_score,
                    active_pixel_ratio=active_pixel_ratio,
                    activity_grid=activity_grid,
                    embedded_segmentation=embedded_segmentation,
                    segmentation_available=seg_available,
                    segmentation_timestamp_us=seg_timestamp,
                    segmentation_time_delta_us=seg_delta,
                    segmentation_relpath=seg_relpath,
                )
                if pbar is not None:
                    pbar.update(1)

        if mp4_writer is not None:
            mp4_writer.close()
            writer.h5f.attrs["companion_mp4_fps"] = float(mp4_writer.resolved_fps or 0.0)
            writer.h5f.attrs["companion_mp4_fps_source"] = str(mp4_writer.fps_source)
            os.replace(tmp_mp4_path, mp4_path)
            mp4_writer = None
        writer.close()
        writer = None
        os.replace(tmp_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if mp4_writer is not None:
            mp4_writer.close()
        cleanup_tmp_file(tmp_path=tmp_path, context=f"exception cleanup for {input_path}", strict=False)
        if save_mp4:
            cleanup_tmp_file(tmp_path=tmp_mp4_path, context=f"exception MP4 cleanup for {input_path}", strict=False)
        raise
    finally:
        if pbar is not None:
            pbar.close()


def _process_file_with_retry(
    input_path: Path,
    output_path: Path,
    height: int,
    width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    window_mode: str,
    image_timestamps_path: Path | None,
    normalize: bool,
    output_dtype: str,
    use_trilinear: bool,
    representation: str,
    event_image_percentile: float,
    save_mp4: bool,
    mp4_fps: float | None,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
    sync_segmentation: bool,
    segmentation_dir: Path | None,
    segmentation_tolerance_us: int,
    tmp_suffix: str,
    split_chunk_duration_s: float | None,
    split_output_path: Path | None,
    split_copy_batch_size: int,
    split_min_windows_per_chunk: int,
    split_chunk_index_pad: int,
    split_metadata_mode: str,
    split_progress_interval_s: float,
    split_log_chunk_progress: bool,
    split_log_dataset_progress: bool,
    split_delete_source_after_success: bool,
    overwrite: bool,
) -> tuple[bool, str | None]:
    stale_tmp_path = tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    if not cleanup_tmp_file(tmp_path=stale_tmp_path, context=f"resume prep for {input_path}", strict=False):
        return False, f"could not remove stale tmp file: {stale_tmp_path}"

    if (
        split_chunk_duration_s is not None
        and float(split_chunk_duration_s) > 0
        and split_output_path is not None
        and output_path.exists()
        and not bool(overwrite)
    ):
        try:
            split_voxel_h5_file(
                input_path=output_path,
                output_base_path=split_output_path,
                chunk_duration_s=float(split_chunk_duration_s),
                copy_batch_size=int(split_copy_batch_size),
                min_windows_per_chunk=int(split_min_windows_per_chunk),
                chunk_index_pad=int(split_chunk_index_pad),
                overwrite=False,
                metadata_mode=str(split_metadata_mode),
                progress_interval_s=float(split_progress_interval_s),
                log_chunk_progress=bool(split_log_chunk_progress),
                log_dataset_progress=bool(split_log_dataset_progress),
                delete_source_companion_mp4=bool(split_delete_source_after_success),
            )
            if bool(split_delete_source_after_success):
                output_path.unlink(missing_ok=True)
            return True, None
        except Exception as exc:
            return False, str(exc)

    for attempt in (1, 2):
        try:
            process_single_file(
                input_path=input_path,
                output_path=output_path,
                height=height,
                width=width,
                downsample_factor=downsample_factor,
                t_bins=t_bins,
                split_polarity=split_polarity,
                accum_time=accum_time,
                stride_time=stride_time,
                start_time_us=start_time_us,
                window_mode=window_mode,
                image_timestamps_path=image_timestamps_path,
                normalize=normalize,
                output_dtype=output_dtype,
                use_trilinear=use_trilinear,
                representation=representation,
                event_image_percentile=event_image_percentile,
                save_mp4=save_mp4,
                mp4_fps=mp4_fps,
                activity_mode=activity_mode,
                activity_spatial_patch_size=activity_spatial_patch_size,
                activity_temporal_patch_size=activity_temporal_patch_size,
                sync_segmentation=sync_segmentation,
                segmentation_dir=segmentation_dir,
                segmentation_tolerance_us=segmentation_tolerance_us,
                show_pbar=False,
                tmp_suffix=tmp_suffix,
            )
            if split_chunk_duration_s is not None and float(split_chunk_duration_s) > 0:
                if split_output_path is None:
                    raise ValueError("split_output_path must be provided when split_chunk_duration_s is set")
                split_voxel_h5_file(
                    input_path=output_path,
                    output_base_path=split_output_path,
                    chunk_duration_s=float(split_chunk_duration_s),
                    copy_batch_size=int(split_copy_batch_size),
                    min_windows_per_chunk=int(split_min_windows_per_chunk),
                    chunk_index_pad=int(split_chunk_index_pad),
                    overwrite=bool(overwrite),
                    metadata_mode=str(split_metadata_mode),
                    progress_interval_s=float(split_progress_interval_s),
                    log_chunk_progress=bool(split_log_chunk_progress),
                    log_dataset_progress=bool(split_log_dataset_progress),
                    delete_source_companion_mp4=bool(split_delete_source_after_success),
                )
                if bool(split_delete_source_after_success):
                    output_path.unlink(missing_ok=True)
            return True, None
        except Exception as exc:
            if attempt == 1:
                cleanup_ok = cleanup_tmp_file(
                    tmp_path=stale_tmp_path,
                    context=f"retry prep for {input_path}",
                    strict=False,
                )
                if not cleanup_ok:
                    return False, f"retry cleanup failed for {stale_tmp_path}: {exc}"
                continue
            return False, str(exc)

    return False, "unknown failure"


def _worker_process_file(job: dict) -> tuple[str, bool, str | None]:
    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])
    image_timestamps_path = Path(job["image_timestamps_path"]) if job.get("image_timestamps_path") else None
    segmentation_dir = Path(job["segmentation_dir"]) if job.get("segmentation_dir") else None
    ok, err = _process_file_with_retry(
        input_path=input_path,
        output_path=output_path,
        height=job["height"],
        width=job["width"],
        downsample_factor=job["downsample_factor"],
        t_bins=job["t_bins"],
        split_polarity=job["split_polarity"],
        accum_time=job["accum_time"],
        stride_time=job["stride_time"],
        start_time_us=job["start_time_us"],
        window_mode=job["window_mode"],
        image_timestamps_path=image_timestamps_path,
        normalize=job["normalize"],
        output_dtype=job["output_dtype"],
        use_trilinear=job["use_trilinear"],
        representation=job["representation"],
        event_image_percentile=job["event_image_percentile"],
        save_mp4=job["save_mp4"],
        mp4_fps=job["mp4_fps"],
        activity_mode=job["activity_mode"],
        activity_spatial_patch_size=job["activity_spatial_patch_size"],
        activity_temporal_patch_size=job["activity_temporal_patch_size"],
        sync_segmentation=job["sync_segmentation"],
        segmentation_dir=segmentation_dir,
        segmentation_tolerance_us=job["segmentation_tolerance_us"],
        tmp_suffix=job["tmp_suffix"],
        split_chunk_duration_s=job["split_chunk_duration_s"],
        split_output_path=None if job["split_output_path"] is None else Path(job["split_output_path"]),
        split_copy_batch_size=job["split_copy_batch_size"],
        split_min_windows_per_chunk=job["split_min_windows_per_chunk"],
        split_chunk_index_pad=job["split_chunk_index_pad"],
        split_metadata_mode=job["split_metadata_mode"],
        split_progress_interval_s=job["split_progress_interval_s"],
        split_log_chunk_progress=job["split_log_chunk_progress"],
        split_log_dataset_progress=job["split_log_dataset_progress"],
        split_delete_source_after_success=job["split_delete_source_after_success"],
        overwrite=job["overwrite"],
    )
    return str(input_path), ok, err


def find_dsec_event_files(dsec_root: Path, splits: list[str]) -> list[dict]:
    records: list[dict] = []
    seen_paths: set[Path] = set()

    for split in splits:
        base_dirs = [
            (dsec_root / split, "split"),
            (dsec_root / f"{split}_events", "split_events"),
        ]
        found_any_base = False
        for base_dir, layout in base_dirs:
            if not base_dir.exists():
                continue
            found_any_base = True

            for sequence_dir in sorted(base_dir.iterdir()):
                if not sequence_dir.is_dir():
                    continue
                input_file = sequence_dir / "events/left/events.h5"
                if not input_file.exists():
                    print(f"[WARN] missing events file: {input_file}")
                    continue
                resolved = input_file.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                records.append(
                    {
                        "input_path": input_file,
                        "split": split,
                        "sequence": sequence_dir.name,
                        "layout": layout,
                    }
                )

        if not found_any_base:
            print(f"[WARN] missing split directory for '{split}': {dsec_root / split} or {dsec_root / f'{split}_events'}")

    records.sort(key=lambda r: str(r["input_path"]))
    return records


def _resolve_image_timestamps_path(
    input_path: Path,
    split: str,
    sequence: str,
    dataset_root: Path,
    image_root: Path | None,
) -> Path:
    sequence_dir = input_path.parent.parent.parent
    base = image_root if image_root is not None else dataset_root

    candidates = [
        sequence_dir / "images/timestamps.txt",
        sequence_dir / "images/left/timestamps.txt",
        base / split / sequence / "images/timestamps.txt",
        base / split / sequence / "images/left/timestamps.txt",
        base / f"{split}_images" / sequence / "images/timestamps.txt",
        base / f"{split}_images" / sequence / "images/left/timestamps.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"could not resolve image timestamps for split='{split}', sequence='{sequence}'. tried:\n{tried}"
    )


def _resolve_segmentation_dir(
    input_path: Path,
    split: str,
    sequence: str,
    dataset_root: Path,
    segmentation_root: Path | None,
    segmentation_subdir: str,
) -> Path | None:
    sequence_dir = input_path.parent.parent.parent
    base = segmentation_root if segmentation_root is not None else dataset_root

    candidates = [
        # Sequence-local layouts
        sequence_dir / "semantic" / segmentation_subdir,
        sequence_dir / "semantic_segmentation" / segmentation_subdir,
        # Common split/sequence layouts under base root
        base / split / sequence / "semantic" / segmentation_subdir,
        base / split / sequence / "semantic_segmentation" / segmentation_subdir,
        base / split / sequence / segmentation_subdir,
        # Global semantic directories
        base / "semantic" / split / sequence / segmentation_subdir,
        base / "semantic" / sequence / segmentation_subdir,
        base / "semantic_segmentation" / split / sequence / segmentation_subdir,
        base / "semantic_segmentation" / sequence / segmentation_subdir,
        base / f"{split}_semantic_segmentation" / split / sequence / segmentation_subdir,
        base / f"{split}_semantic_segmentation" / sequence / segmentation_subdir,
    ]

    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _build_output_path(
    input_path: Path,
    dsec_root: Path,
    output_root: Path | None,
    output_name: str | None,
    normalized_output_suffix: str | None,
    normalized_subdir: str | None,
    downsample_factor: int,
) -> Path:
    if normalized_output_suffix is not None:
        output_filename = f"{input_path.stem}{normalized_output_suffix}"
        output_filename = ensure_scale_tag_in_filename(output_filename, downsample_factor=downsample_factor)
    elif output_name is not None:
        output_filename = output_name
    else:
        raise ValueError("either output_name or output_suffix must be provided")

    if output_root is None:
        output_dir = input_path.parent
    else:
        rel_input = input_path.relative_to(dsec_root)
        output_dir = output_root / rel_input.parent

    if normalized_subdir is not None:
        output_dir = output_dir / normalized_subdir
    return output_dir / output_filename


def _build_split_output_path(
    output_path: Path,
    dataset_root: Path,
    output_root: Path | None,
    split_output_root: Path | None,
) -> Path:
    if split_output_root is None:
        return output_path
    if output_root is not None:
        relative_output = output_path.relative_to(output_root)
    else:
        relative_output = output_path.relative_to(dataset_root)
    return split_output_root / relative_output


def _split_chunk_outputs_exist(output_base_path: Path) -> bool:
    pattern = f"{output_base_path.stem}_part*{output_base_path.suffix}"
    return any(output_base_path.parent.glob(pattern))


def process_dataset_root(
    dataset_root: Path,
    splits: list[str],
    output_name: str | None,
    overwrite: bool,
    output_root: Path | None,
    height: int,
    width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    window_mode: str,
    image_root: Path | None,
    normalize: bool,
    output_dtype: str,
    use_trilinear: bool,
    representation: str,
    event_image_percentile: float,
    save_mp4: bool,
    mp4_fps: float | None,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
    sync_segmentation: bool,
    segmentation_root: Path | None,
    segmentation_subdir: str,
    segmentation_tolerance_us: int,
    tmp_suffix: str,
    num_processes: int,
    split_chunk_duration_s: float | None,
    split_output_root: Path | None,
    split_copy_batch_size: int,
    split_min_windows_per_chunk: int,
    split_chunk_index_pad: int,
    split_metadata_mode: str,
    split_progress_interval_s: float,
    split_log_chunk_progress: bool,
    split_log_dataset_progress: bool,
    split_delete_source_after_success: bool,
    output_suffix: str | None = None,
    output_subdir: str | None = None,
    start_time_us: int | None = None,
):
    if int(num_processes) < 1:
        raise ValueError("num_processes must be >= 1")
    if save_mp4 and representation != "event_image":
        raise ValueError("save_mp4 requires representation=event_image")
    if output_name is not None and output_suffix is not None:
        raise ValueError("use either output_name or output_suffix, not both")
    if representation not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported representation: {representation}")
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")
    if split_chunk_duration_s is not None and float(split_chunk_duration_s) <= 0:
        raise ValueError("split_chunk_duration_s must be > 0 when provided")
    if int(split_copy_batch_size) < 1:
        raise ValueError("split_copy_batch_size must be >= 1")
    if int(split_min_windows_per_chunk) < 1:
        raise ValueError("split_min_windows_per_chunk must be >= 1")
    if int(split_chunk_index_pad) < 1:
        raise ValueError("split_chunk_index_pad must be >= 1")
    if str(split_metadata_mode) not in {"full", "minimal"}:
        raise ValueError("split_metadata_mode must be one of {'full', 'minimal'}")
    if float(split_progress_interval_s) < 0:
        raise ValueError("split_progress_interval_s must be >= 0")

    normalized_suffix = normalized_output_suffix(output_suffix) if output_suffix is not None else None
    normalized_subdir = normalized_output_subdir(output_subdir)
    resolved_output_name = (
        output_name
        if output_name is not None
        else ensure_scale_tag_in_filename("voxels.h5", downsample_factor=downsample_factor)
    )

    input_records = find_dsec_event_files(dsec_root=dataset_root, splits=splits)
    if len(input_records) == 0:
        raise FileNotFoundError(f"No events.h5 found under root={dataset_root}, splits={splits}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    if split_output_root is not None:
        split_output_root.mkdir(parents=True, exist_ok=True)

    if sync_segmentation:
        num_with_semantic = 0
        num_without_semantic = 0
        by_split = {}

        for record in input_records:
            seg_dir = _resolve_segmentation_dir(
                input_path=Path(record["input_path"]),
                split=str(record["split"]),
                sequence=str(record["sequence"]),
                dataset_root=dataset_root,
                segmentation_root=segmentation_root,
                segmentation_subdir=segmentation_subdir,
            )

            split = str(record["split"])
            if split not in by_split:
                by_split[split] = {"with": 0, "without": 0}

            if seg_dir is not None:
                num_with_semantic += 1
                by_split[split]["with"] += 1
            else:
                num_without_semantic += 1
                by_split[split]["without"] += 1

        print("[SEGMENTATION STATS]")
        print(
            f"total={len(input_records)}, "
            f"with_semantic={num_with_semantic}, "
            f"without_semantic={num_without_semantic}"
        )

        for split in sorted(by_split.keys()):
            s = by_split[split]
            print(
                f"  split={split}: "
                f"with={s['with']}, without={s['without']}"
            )

    jobs: list[dict] = []
    num_done = 0
    num_skipped = 0
    num_skipped_missing_segmentation = 0
    num_failed = 0

    for record in tqdm.tqdm(input_records, desc="DSEC sequences"):
        input_path = Path(record["input_path"])
        output_path = _build_output_path(
            input_path=input_path,
            dsec_root=dataset_root,
            output_root=output_root,
            output_name=resolved_output_name,
            normalized_output_suffix=normalized_suffix,
            normalized_subdir=normalized_subdir,
            downsample_factor=downsample_factor,
        )
        split_output_path = None
        if split_chunk_duration_s is not None and float(split_chunk_duration_s) > 0:
            split_output_path = _build_split_output_path(
                output_path=output_path,
                dataset_root=dataset_root,
                output_root=output_root,
                split_output_root=split_output_root,
            )
            if (
                not output_path.exists()
                and not overwrite
                and bool(split_delete_source_after_success)
                and split_output_path is not None
                and _split_chunk_outputs_exist(split_output_path)
            ):
                num_skipped += 1
                continue
        if output_path.exists():
            if overwrite:
                output_path.unlink()
            elif split_output_path is None:
                num_skipped += 1
                continue

        image_timestamps_path: Path | None = None
        if window_mode == "image_middle":
            try:
                image_timestamps_path = _resolve_image_timestamps_path(
                    input_path=input_path,
                    split=str(record["split"]),
                    sequence=str(record["sequence"]),
                    dataset_root=dataset_root,
                    image_root=image_root,
                )
            except Exception as exc:
                num_failed += 1
                print(f"[FAILED] {input_path}: {exc}")
                continue

        segmentation_dir: Path | None = None
        if sync_segmentation:
            segmentation_dir = _resolve_segmentation_dir(
                input_path=input_path,
                split=str(record["split"]),
                sequence=str(record["sequence"]),
                dataset_root=dataset_root,
                segmentation_root=segmentation_root,
                segmentation_subdir=segmentation_subdir,
            )
            if segmentation_dir is None:
                num_skipped += 1
                num_skipped_missing_segmentation += 1
                print(
                    "[SKIP] segmentation directory not found for semantic preprocessing: "
                    f"split={record['split']}, sequence={record['sequence']}"
                )
                continue

        jobs.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "height": int(height),
                "width": int(width),
                "downsample_factor": int(downsample_factor),
                "t_bins": int(t_bins),
                "split_polarity": bool(split_polarity),
                "accum_time": int(accum_time),
                "stride_time": int(stride_time),
                "start_time_us": int(start_time_us) if start_time_us is not None else None,
                "window_mode": window_mode,
                "image_timestamps_path": str(image_timestamps_path) if image_timestamps_path is not None else None,
                "normalize": bool(normalize),
                "output_dtype": output_dtype,
                "use_trilinear": bool(use_trilinear),
                "representation": str(representation),
                "event_image_percentile": float(event_image_percentile),
                "save_mp4": bool(save_mp4),
                "mp4_fps": None if mp4_fps is None else float(mp4_fps),
                "activity_mode": str(activity_mode),
                "activity_spatial_patch_size": int(activity_spatial_patch_size),
                "activity_temporal_patch_size": int(activity_temporal_patch_size),
                "sync_segmentation": bool(sync_segmentation),
                "segmentation_dir": str(segmentation_dir) if segmentation_dir is not None else None,
                "segmentation_tolerance_us": int(segmentation_tolerance_us),
                "tmp_suffix": tmp_suffix,
                "split_chunk_duration_s": split_chunk_duration_s,
                "split_output_path": None if split_output_path is None else str(split_output_path),
                "split_copy_batch_size": int(split_copy_batch_size),
                "split_min_windows_per_chunk": int(split_min_windows_per_chunk),
                "split_chunk_index_pad": int(split_chunk_index_pad),
                "split_metadata_mode": str(split_metadata_mode),
                "split_progress_interval_s": float(split_progress_interval_s),
                "split_log_chunk_progress": bool(split_log_chunk_progress),
                "split_log_dataset_progress": bool(split_log_dataset_progress),
                "split_delete_source_after_success": bool(split_delete_source_after_success),
                "overwrite": bool(overwrite),
            }
        )

    if len(jobs) > 0:
        if int(num_processes) == 1:
            iterator = (_worker_process_file(job) for job in jobs)
            for input_name, success, err in tqdm.tqdm(iterator, total=len(jobs), desc="DSEC workers"):
                if success:
                    num_done += 1
                else:
                    num_failed += 1
                    print(f"[FAILED] {input_name}: {err}")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(num_processes)) as pool:
                for input_name, success, err in tqdm.tqdm(
                    pool.imap_unordered(_worker_process_file, jobs),
                    total=len(jobs),
                    desc="DSEC workers",
                ):
                    if success:
                        num_done += 1
                    else:
                        num_failed += 1
                        print(f"[FAILED] {input_name}: {err}")

    print(
        "[SUMMARY] "
        f"done={num_done}, skipped={num_skipped}, "
        f"skipped_missing_segmentation={num_skipped_missing_segmentation}, failed={num_failed}"
    )
    if num_failed > 0:
        raise RuntimeError(f"{num_failed} sequences failed while processing {dataset_root}")


def process_dsec_root(
    dsec_root: Path,
    splits: list[str],
    output_name: str | None,
    overwrite: bool,
    output_root: Path | None,
    height: int,
    width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    window_mode: str,
    image_root: Path | None,
    normalize: bool,
    output_dtype: str,
    use_trilinear: bool,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
    sync_segmentation: bool,
    segmentation_root: Path | None,
    segmentation_subdir: str,
    segmentation_tolerance_us: int,
    tmp_suffix: str,
    num_processes: int,
    split_chunk_duration_s: float | None = None,
    split_output_root: Path | None = None,
    split_copy_batch_size: int = 8,
    split_min_windows_per_chunk: int = 1,
    split_chunk_index_pad: int = 4,
    split_metadata_mode: str = "full",
    split_progress_interval_s: float = 0.0,
    split_log_chunk_progress: bool = False,
    split_log_dataset_progress: bool = False,
    split_delete_source_after_success: bool = False,
    output_suffix: str | None = None,
    output_subdir: str | None = None,
    start_time_us: int | None = None,
):
    # Backward-compatible wrapper.
    process_dataset_root(
        dataset_root=dsec_root,
        splits=splits,
        output_name=output_name,
        output_suffix=output_suffix,
        output_subdir=output_subdir,
        overwrite=overwrite,
        output_root=output_root,
        height=height,
        width=width,
        downsample_factor=downsample_factor,
        t_bins=t_bins,
        split_polarity=split_polarity,
        accum_time=accum_time,
        stride_time=stride_time,
        start_time_us=start_time_us,
        window_mode=window_mode,
        image_root=image_root,
        normalize=normalize,
        output_dtype=output_dtype,
        use_trilinear=use_trilinear,
        activity_mode=activity_mode,
        activity_spatial_patch_size=activity_spatial_patch_size,
        activity_temporal_patch_size=activity_temporal_patch_size,
        sync_segmentation=sync_segmentation,
        segmentation_root=segmentation_root,
        segmentation_subdir=segmentation_subdir,
        segmentation_tolerance_us=segmentation_tolerance_us,
        tmp_suffix=tmp_suffix,
        num_processes=num_processes,
        split_chunk_duration_s=split_chunk_duration_s,
        split_output_root=split_output_root,
        split_copy_batch_size=split_copy_batch_size,
        split_min_windows_per_chunk=split_min_windows_per_chunk,
        split_chunk_index_pad=split_chunk_index_pad,
        split_metadata_mode=split_metadata_mode,
        split_progress_interval_s=split_progress_interval_s,
        split_log_chunk_progress=split_log_chunk_progress,
        split_log_dataset_progress=split_log_dataset_progress,
        split_delete_source_after_success=split_delete_source_after_success,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Build event representations from DSEC events.h5 files.""")
    parser.add_argument("--input_path", type=Path, help="Path to input events.h5.")
    parser.add_argument("--output_path", type=Path, help="Path where output voxel .h5 will be written.")
    parser.add_argument("--dsec_root", type=Path, help="Path to DSEC root (contains train/test splits).")
    parser.add_argument("--dataset_root", type=Path, help="Alias of --dsec_root for naming consistency.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Split names for root mode.")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Deprecated: exact output filename per sequence in root mode (e.g. voxels.h5).",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Output suffix in root mode (e.g. _voxels.h5). Scale tag (_1x/_2x) is auto-added.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Optional output subdirectory under each target dir (e.g. voxel).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional output root for root mode (preserves relative split paths).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in root mode.",
    )
    parser.add_argument(
        "--tmp_suffix",
        type=str,
        default=".tmp",
        help="Temporary suffix used while writing (renamed on success).",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Parallel workers for root mode (spawn).",
    )
    parser.add_argument(
        "--split_chunk_duration_s",
        type=float,
        default=None,
        help="Optional: after writing each voxel H5, immediately split it into duration-based chunks.",
    )
    parser.add_argument(
        "--split_output_root",
        type=Path,
        default=None,
        help="Optional output root for split chunk files. Default writes chunks alongside the unsplit output.",
    )
    parser.add_argument("--split_copy_batch_size", type=int, default=8, help="Split row-copy batch size.")
    parser.add_argument(
        "--split_min_windows_per_chunk",
        type=int,
        default=1,
        help="Drop split chunks with fewer than this number of windows.",
    )
    parser.add_argument(
        "--split_chunk_index_pad",
        type=int,
        default=4,
        help="Zero-padding width for split chunk suffix `_partXXXX`.",
    )
    parser.add_argument(
        "--split_metadata_mode",
        choices=["full", "minimal"],
        default="full",
        help="Metadata copy mode for split chunk files.",
    )
    parser.add_argument(
        "--split_progress_interval_s",
        type=float,
        default=0.0,
        help="If >0, print progress every N seconds while copying split chunk rows.",
    )
    parser.add_argument(
        "--split_log_chunk_progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print split chunk start/end logs.",
    )
    parser.add_argument(
        "--split_log_dataset_progress",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print split dataset copy logs.",
    )
    parser.add_argument(
        "--split_delete_source_after_success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete the unsplit voxel H5 after split chunks are successfully created.",
    )
    parser.add_argument(
        "--activity_mode",
        choices=["full", "light"],
        default="full",
        help="Activity metadata layout saved per window.",
    )
    parser.add_argument("--activity_spatial_patch_size", type=int, default=16, help="Spatial patch size for activity metadata.")
    parser.add_argument("--activity_temporal_patch_size", type=int, default=2, help="Temporal patch size for full activity metadata.")
    parser.add_argument("--input_height", type=int, default=480, help="Input event height (default: 480).")
    parser.add_argument("--input_width", type=int, default=640, help="Input event width (default: 640).")
    parser.add_argument("--height", dest="input_height", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--width", dest="input_width", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--downsample_factor",
        type=int,
        choices=[1, 2],
        default=1,
        help="Nearest-neighbor spatial downsample factor (2 means 1/2 resolution).",
    )
    parser.add_argument(
        "--representation",
        choices=["voxel_grid", "event_image"],
        default="voxel_grid",
        help="Output representation. event_image stores one 3-channel red/blue image per window.",
    )
    parser.add_argument(
        "--event_image_percentile",
        type=float,
        default=99.0,
        help="Percentile clip used when normalizing event_image counts; ignored for voxel_grid.",
    )
    parser.add_argument(
        "--save_mp4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When representation=event_image, also save a companion MP4 beside each output H5.",
    )
    parser.add_argument(
        "--mp4_fps",
        type=float,
        default=None,
        help="Optional FPS for companion MP4 export. If omitted, infer from timestamp spacing.",
    )
    parser.add_argument(
        "--t_bins",
        type=int,
        default=10,
        help="Number of temporal bins for voxel_grid. Ignored for event_image.",
    )
    parser.add_argument(
        "--split_polarity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate positive/negative events into separate channels (output channels = 2*t_bins).",
    )
    parser.add_argument(
        "--accum_time",
        type=int,
        default=50000,
        help="Accumulation window in microseconds.",
    )
    parser.add_argument(
        "--stride_time",
        type=int,
        default=None,
        help="Sliding stride in microseconds (default: same as --accum_time).",
    )
    parser.add_argument(
        "--start_time_us",
        type=int,
        default=None,
        help="Optional fixed time origin in us for fixed windows.",
    )
    parser.add_argument(
        "--window_mode",
        choices=["fixed", "image_middle"],
        default="fixed",
        help="Window policy: fixed stride windows or midpoint windows from image timestamps.",
    )
    parser.add_argument(
        "--image_timestamps_path",
        type=Path,
        default=None,
        help="Path to timestamps.txt used when --window_mode image_middle in single-file mode.",
    )
    parser.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help="Optional root used to resolve timestamps.txt in root mode.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize non-zero voxel values per sample (voxel_grid only).",
    )
    parser.add_argument(
        "--output_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Stored dtype for the output representation tensor in HDF5.",
    )
    parser.add_argument(
        "--use_trilinear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trilinear interpolation in voxelization (voxel_grid only).",
    )
    parser.add_argument(
        "--sync_segmentation",
        action="store_true",
        help="Store segmentation synchronization metadata per voxel window.",
    )
    parser.add_argument(
        "--segmentation_dir",
        type=Path,
        default=None,
        help="Single-file mode label directory containing <timestamp>.png files.",
    )
    parser.add_argument(
        "--segmentation_root",
        type=Path,
        default=None,
        help="Root mode label root used for per-sequence segmentation lookup.",
    )
    parser.add_argument(
        "--segmentation_subdir",
        type=str,
        default="11classes",
        help="Segmentation subdirectory name under each sequence.",
    )
    parser.add_argument(
        "--segmentation_tolerance_us",
        type=int,
        default=0,
        help="Timestamp tolerance used for segmentation matching.",
    )
    args = parser.parse_args()

    stride_time = args.accum_time if args.stride_time is None else args.stride_time

    is_single_mode = args.input_path is not None or args.output_path is not None
    root_dir = args.dataset_root if args.dataset_root is not None else args.dsec_root
    is_root_mode = root_dir is not None

    if is_single_mode and is_root_mode:
        parser.error(
            "Use either single-file mode (--input_path/--output_path) "
            "or root mode (--dataset_root/--dsec_root), not both."
        )

    if is_root_mode:
        if args.output_name is not None and args.output_suffix is not None:
            parser.error("Use either --output_name or --output_suffix, not both.")
        process_dataset_root(
            dataset_root=root_dir,
            splits=args.splits,
            output_name=args.output_name,
            output_suffix=args.output_suffix,
            output_subdir=args.output_subdir,
            overwrite=args.overwrite,
            output_root=args.output_root,
            height=args.input_height,
            width=args.input_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            split_polarity=args.split_polarity,
            accum_time=args.accum_time,
            stride_time=stride_time,
            start_time_us=args.start_time_us,
            window_mode=args.window_mode,
            image_root=args.image_root,
            normalize=args.normalize,
            output_dtype=args.output_dtype,
            use_trilinear=args.use_trilinear,
            representation=args.representation,
            event_image_percentile=args.event_image_percentile,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
            activity_mode=args.activity_mode,
            activity_spatial_patch_size=args.activity_spatial_patch_size,
            activity_temporal_patch_size=args.activity_temporal_patch_size,
            sync_segmentation=args.sync_segmentation,
            segmentation_root=args.segmentation_root,
            segmentation_subdir=args.segmentation_subdir,
            segmentation_tolerance_us=args.segmentation_tolerance_us,
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
            split_chunk_duration_s=args.split_chunk_duration_s,
            split_output_root=args.split_output_root,
            split_copy_batch_size=args.split_copy_batch_size,
            split_min_windows_per_chunk=args.split_min_windows_per_chunk,
            split_chunk_index_pad=args.split_chunk_index_pad,
            split_metadata_mode=args.split_metadata_mode,
            split_progress_interval_s=args.split_progress_interval_s,
            split_log_chunk_progress=args.split_log_chunk_progress,
            split_log_dataset_progress=args.split_log_dataset_progress,
            split_delete_source_after_success=args.split_delete_source_after_success,
        )
    else:
        if args.input_path is None or args.output_path is None:
            parser.error("Single-file mode requires both --input_path and --output_path.")
        if args.window_mode == "image_middle" and args.image_timestamps_path is None:
            parser.error("Single-file mode with --window_mode image_middle requires --image_timestamps_path.")

        process_single_file(
            input_path=args.input_path,
            output_path=args.output_path,
            height=args.input_height,
            width=args.input_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            split_polarity=args.split_polarity,
            accum_time=args.accum_time,
            stride_time=stride_time,
            start_time_us=args.start_time_us,
            window_mode=args.window_mode,
            image_timestamps_path=args.image_timestamps_path,
            normalize=args.normalize,
            output_dtype=args.output_dtype,
            use_trilinear=args.use_trilinear,
            representation=args.representation,
            event_image_percentile=args.event_image_percentile,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
            activity_mode=args.activity_mode,
            activity_spatial_patch_size=args.activity_spatial_patch_size,
            activity_temporal_patch_size=args.activity_temporal_patch_size,
            sync_segmentation=args.sync_segmentation,
            segmentation_dir=args.segmentation_dir,
            segmentation_tolerance_us=args.segmentation_tolerance_us,
            show_pbar=True,
            tmp_suffix=args.tmp_suffix,
        )
