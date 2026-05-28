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

DEFAULT_COMPRESSION_LEVEL = 1
H5_COMPRESSION_FLAGS = get_h5_compression_flags(compression_level=DEFAULT_COMPRESSION_LEVEL)
MS_TO_IDX_BUILD_CHUNK_EVENTS = 5_000_000
ACTIVITY_MODES = {"full", "light"}
REPRESENTATION_MODES = {"voxel_grid", "event_image"}


def _configure_h5_compression(compression_level: int) -> None:
    global H5_COMPRESSION_FLAGS
    H5_COMPRESSION_FLAGS = get_h5_compression_flags(compression_level=int(compression_level))


def _read_t_offset(filehandle: h5py.File) -> int:
    if "t_offset" not in filehandle:
        return 0
    return int(filehandle["t_offset"][()])


def get_num_events(h5file: Path) -> int:
    with h5py.File(str(h5file), "r") as h5f:
        return len(h5f["events/t"])


def get_resolution(h5file: Path) -> tuple[int | None, int | None]:
    with h5py.File(str(h5file), "r") as h5f:
        events = h5f["events"]
        if "height" in events and "width" in events:
            height = int(events["height"][()])
            width = int(events["width"][()])
            return height, width
    return None, None


def _empty_events() -> dict[str, np.ndarray]:
    return {
        "x": np.empty((0,), dtype=np.uint16),
        "y": np.empty((0,), dtype=np.uint16),
        "p": np.empty((0,), dtype=np.int16),
        "t": np.empty((0,), dtype=np.int64),
    }


def _extract_from_h5_by_index(filehandle: h5py.File, ev_start_idx: int, ev_end_idx: int) -> dict[str, np.ndarray]:
    events = filehandle["events"]
    t_offset = _read_t_offset(filehandle)
    return {
        "x": np.asarray(events["x"][ev_start_idx:ev_end_idx]),
        "y": np.asarray(events["y"][ev_start_idx:ev_end_idx]),
        "p": np.asarray(events["p"][ev_start_idx:ev_end_idx]),
        "t": np.asarray(events["t"][ev_start_idx:ev_end_idx], dtype=np.int64) + t_offset,
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
    start_us: int,
    end_us: int,
    time_offset_us: int = 0,
) -> tuple[int, int]:
    if ms_to_idx is None or ms_to_idx.size == 0:
        return 0, num_events

    rel_start_us = int(start_us) - int(time_offset_us)
    rel_end_us = int(end_us) - int(time_offset_us)
    if rel_end_us <= 0:
        return 0, 0

    start_ms = max(int(rel_start_us // 1000), 0)
    end_ms_exclusive = max(int((rel_end_us + 999) // 1000), start_ms + 1)

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
        start_us=start_us,
        end_us=end_us,
        time_offset_us=t_offset,
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

    # Fast path: avoid reading events/t twice.
    # We already loaded coarse t range above; reuse relative bounds for x/y/p/t slices.
    events = filehandle["events"]
    x_coarse = np.asarray(events["x"][coarse_start:coarse_end])
    y_coarse = np.asarray(events["y"][coarse_start:coarse_end])
    p_coarse = np.asarray(events["p"][coarse_start:coarse_end])
    return {
        "x": x_coarse[rel_start:rel_end],
        "y": y_coarse[rel_start:rel_end],
        "p": p_coarse[rel_start:rel_end],
        "t": t_coarse_abs[rel_start:rel_end],
    }


def _resolve_input_resolution(input_path: Path, input_height: int | None, input_width: int | None) -> tuple[int, int]:
    auto_height, auto_width = get_resolution(input_path)
    resolved_height = input_height if input_height is not None else auto_height
    resolved_width = input_width if input_width is not None else auto_width

    # Fallback to common 1MP settings if metadata is unavailable.
    if resolved_height is None:
        resolved_height = 720
    if resolved_width is None:
        resolved_width = 1280
    return int(resolved_height), int(resolved_width)


def _resolve_output_resolution(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    downsample_factor: int,
) -> tuple[int, int]:
    if int(downsample_factor) < 1:
        raise ValueError("downsample_factor must be >= 1")
    if int(downsample_factor) == 1:
        return int(output_height), int(output_width)

    if input_height % int(downsample_factor) != 0 or input_width % int(downsample_factor) != 0:
        raise ValueError(
            "input resolution must be divisible by downsample_factor for nearest downsample: "
            f"{input_width}x{input_height}, factor={downsample_factor}"
        )
    return int(input_height // int(downsample_factor)), int(input_width // int(downsample_factor))


def _spatially_normalize_events(
    events: dict[str, np.ndarray],
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
) -> dict[str, np.ndarray]:
    if len(events["t"]) == 0:
        return _empty_events()

    x_src = events["x"].astype(np.float32, copy=False)
    y_src = events["y"].astype(np.float32, copy=False)
    p_src = events["p"]
    t_src = events["t"]

    valid_src = (
        (x_src >= 0)
        & (x_src < float(input_width))
        & (y_src >= 0)
        & (y_src < float(input_height))
    )
    if not np.any(valid_src):
        return _empty_events()

    x_src = x_src[valid_src]
    y_src = y_src[valid_src]
    p_src = p_src[valid_src]
    t_src = t_src[valid_src]

    if output_width == input_width and output_height == input_height:
        x_out = x_src.astype(np.float32, copy=False)
        y_out = y_src.astype(np.float32, copy=False)
    elif input_width % output_width == 0 and input_height % output_height == 0:
        fx = input_width // output_width
        fy = input_height // output_height
        # Nearest-neighbor resize for integer downsample ratio.
        x_out = np.floor(x_src / float(fx)).astype(np.float32, copy=False)
        y_out = np.floor(y_src / float(fy)).astype(np.float32, copy=False)
    else:
        # Nearest-neighbor resize for generic ratio.
        x_out = np.floor(x_src * (float(output_width) / float(input_width))).astype(np.float32, copy=False)
        y_out = np.floor(y_src * (float(output_height) / float(input_height))).astype(np.float32, copy=False)

    valid_out = (
        (x_out >= 0)
        & (x_out < float(output_width))
        & (y_out >= 0)
        & (y_out < float(output_height))
    )
    if not np.any(valid_out):
        return _empty_events()

    # Normalize both {0,1} and {-1,+1} source conventions to binary {0,1}.
    # EventVoxelGrid applies signed mapping via (2*p-1) internally.
    p_bin = normalize_polarity_to_binary(p_src, dtype=np.float32)

    return {
        "x": x_out[valid_out].astype(np.float32, copy=False),
        "y": y_out[valid_out].astype(np.float32, copy=False),
        "p": p_bin[valid_out].astype(np.float32, copy=False),
        "t": t_src[valid_out].astype(np.int64, copy=False),
    }


def _events_to_voxel_numpy(
    events: dict[str, np.ndarray],
    voxelizer: EventVoxelGrid,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    normalized = _spatially_normalize_events(
        events=events,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
    )
    if len(normalized["t"]) == 0:
        return np.zeros(voxelizer.voxel_grid.shape, dtype=np.float32)

    t_shifted = (normalized["t"] - normalized["t"][0]).astype(np.float32, copy=False)
    events_torch = {
        "x": torch.from_numpy(normalized["x"]),
        "y": torch.from_numpy(normalized["y"]),
        "p": torch.from_numpy(normalized["p"]),
        "t": torch.from_numpy(t_shifted),
    }
    voxel = voxelizer.convert(events_torch)
    return voxel.cpu().numpy().astype(np.float32, copy=False)


def _events_to_event_image_numpy(
    events: dict[str, np.ndarray],
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _spatially_normalize_events(
        events=events,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
    )
    return accumulate_events_to_rgb(
        normalized["x"],
        normalized["y"],
        normalized["p"],
        (output_height, output_width),
        percentile=percentile,
        dtype=np.float32,
    )


def _event_image_chw_to_hwc_uint8(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.ndim != 3 or image_chw.shape[0] != 3:
        raise ValueError(f"expected event image [3,H,W], got shape={image_chw.shape}")
    return np.moveaxis(np.asarray(image_chw), 0, -1)


def _build_windows_from_start(
    start_us: int,
    end_exclusive_us: int,
    accum_time_us: int,
    stride_time_us: int,
) -> list[tuple[int, int, int, int]]:
    if end_exclusive_us <= start_us:
        return []

    starts = np.arange(start_us, end_exclusive_us, stride_time_us, dtype=np.int64)
    windows: list[tuple[int, int, int, int]] = []
    for i, win_start in enumerate(starts):
        win_start_int = int(win_start)
        win_end_int = min(win_start_int + int(accum_time_us), int(end_exclusive_us))
        anchor_int = win_start_int + (win_end_int - win_start_int) // 2
        windows.append((i, win_start_int, win_end_int, anchor_int))
    return windows


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
    if temporal_bins <= 0:
        raise ValueError("temporal_bins must be > 0")
    if spatial_patch_size <= 0 or temporal_patch_size <= 0:
        raise ValueError("activity patch sizes must be > 0")

    channels, height, width = voxel.shape
    if split_polarity:
        if channels != temporal_bins * 2:
            raise ValueError(
                f"Expected channels={temporal_bins * 2} for split polarity, got {channels}"
            )
        activity_volume = np.abs(voxel).reshape(2, temporal_bins, height, width).sum(axis=0)
    else:
        if channels != temporal_bins:
            raise ValueError(f"Expected channels={temporal_bins}, got {channels}")
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
    active_pixel_ratio = float(
        np.count_nonzero(activity_volume.sum(axis=0) > 0) / float(max(1, height * width))
    )
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
    grid = padded.reshape(
        tp,
        temporal_patch_size,
        hp,
        spatial_patch_size,
        wp,
        spatial_patch_size,
    ).sum(axis=(1, 3, 5))
    return grid.astype(np.float16, copy=False), nonzero_voxel_ratio, active_pixel_ratio


class VoxelH5Writer:
    def __init__(
        self,
        outfile: Path,
        t_bins: int,
        height: int,
        width: int,
        voxel_dtype: np.dtype,
        activity_mode: str,
        activity_grid_shape: tuple[int, ...],
        initial_capacity: int = 256,
        capacity_growth: str = "double",
    ):
        if outfile.exists():
            raise FileExistsError(f"output already exists: {outfile}")

        self.h5f = h5py.File(str(outfile), "a")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self._capacity = int(initial_capacity)
        if capacity_growth not in {"double", "exact"}:
            raise ValueError("capacity_growth must be 'double' or 'exact'")
        self._capacity_growth = str(capacity_growth)
        self._num_windows = 0
        self._datasets = (
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
        )

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
            "window_index",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
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
            "window_rel_start_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_rel_end_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
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
            "anchor_rel_timestamp_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
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

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def _ensure_capacity(self, needed: int):
        if needed <= self._capacity:
            return
        if self._capacity_growth == "double":
            new_capacity = self._capacity
            while new_capacity < needed:
                new_capacity *= 2
        else:
            # Compare mode: resize only to exact required size (no geometric growth).
            new_capacity = needed
        for dset_name in self._datasets:
            self.h5f[dset_name].resize(new_capacity, axis=0)
        self._capacity = new_capacity

    def add_window(
        self,
        voxel: np.ndarray,
        window_index: int,
        t_start_us: int,
        t_end_us: int,
        rel_start_us: int,
        rel_end_us: int,
        anchor_timestamp_us: int,
        anchor_rel_timestamp_us: int,
        event_count: int,
        activity_score: float,
        active_pixel_ratio: float,
        activity_grid: np.ndarray,
    ):
        idx = self._num_windows
        self._ensure_capacity(idx + 1)

        self.h5f["voxels"][idx] = voxel
        self.h5f["window_index"][idx] = int(window_index)
        self.h5f["window_t_start_us"][idx] = int(t_start_us)
        self.h5f["window_t_end_us"][idx] = int(t_end_us)
        self.h5f["window_rel_start_us"][idx] = int(rel_start_us)
        self.h5f["window_rel_end_us"][idx] = int(rel_end_us)
        self.h5f["anchor_timestamp_us"][idx] = int(anchor_timestamp_us)
        self.h5f["anchor_rel_timestamp_us"][idx] = int(anchor_rel_timestamp_us)
        self.h5f["window_event_count"][idx] = int(event_count)
        self.h5f["window_activity_score"][idx] = float(activity_score)
        self.h5f["window_active_pixel_ratio"][idx] = float(active_pixel_ratio)
        self.h5f["window_activity_grid"][idx] = activity_grid
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
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    normalize: bool,
    output_dtype: str,
    compression_level: int,
    use_trilinear: bool,
    representation: str,
    event_image_percentile: float,
    save_mp4: bool,
    mp4_fps: float | None,
    writer_capacity_growth: str,
    rdcc_nbytes: int,
    rdcc_nslots: int,
    rdcc_w0: float,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
    show_progress: bool,
    tmp_suffix: str,
) -> None:
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_height/output_width must be > 0")
    if int(downsample_factor) not in (1, 2):
        raise ValueError("downsample_factor must be 1 or 2")
    if t_bins <= 0:
        raise ValueError("t_bins must be > 0")
    if accum_time <= 0:
        raise ValueError("accum_time must be > 0")
    if stride_time <= 0:
        raise ValueError("stride_time must be > 0")
    if int(compression_level) < 0 or int(compression_level) > 9:
        raise ValueError("compression_level must be in [0,9]")
    if writer_capacity_growth not in {"double", "exact"}:
        raise ValueError("writer_capacity_growth must be 'double' or 'exact'")
    if int(rdcc_nbytes) <= 0:
        raise ValueError("rdcc_nbytes must be > 0")
    if int(rdcc_nslots) <= 0:
        raise ValueError("rdcc_nslots must be > 0")
    if float(rdcc_w0) < 0.0 or float(rdcc_w0) > 1.0:
        raise ValueError("rdcc_w0 must be in [0,1]")
    if representation not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported representation: {representation}")
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")

    _configure_h5_compression(compression_level=int(compression_level))

    resolved_input_height, resolved_input_width = _resolve_input_resolution(
        input_path=input_path,
        input_height=input_height,
        input_width=input_width,
    )
    effective_output_height, effective_output_width = _resolve_output_resolution(
        input_height=resolved_input_height,
        input_width=resolved_input_width,
        output_height=output_height,
        output_width=output_width,
        downsample_factor=downsample_factor,
    )
    representation_channels = int(t_bins) * (2 if split_polarity else 1) if representation == "voxel_grid" else 3
    activity_temporal_bins = int(t_bins) if representation == "voxel_grid" else 1
    voxel_dtype = np.float16 if output_dtype == "float16" else np.float32
    activity_grid_shape = (
        (
            (int(activity_temporal_bins) + int(activity_temporal_patch_size) - 1) // int(activity_temporal_patch_size),
            (int(effective_output_height) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
            (int(effective_output_width) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
        )
        if activity_mode == "full"
        else (
            (int(effective_output_height) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
            (int(effective_output_width) + int(activity_spatial_patch_size) - 1) // int(activity_spatial_patch_size),
        )
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
        with h5py.File(
            str(input_path),
            "r",
            rdcc_nbytes=int(rdcc_nbytes),
            rdcc_nslots=int(rdcc_nslots),
            rdcc_w0=float(rdcc_w0),
        ) as h5f:
            t_first, t_last_exclusive = _get_time_bounds_us(h5f)
            ms_to_idx = _load_ms_to_idx(h5f)
            ms_to_idx_source = "input"
            if ms_to_idx is None:
                ms_to_idx = _build_ms_to_idx_from_events_t(
                    filehandle=h5f,
                    chunk_events=MS_TO_IDX_BUILD_CHUNK_EVENTS,
                )
                ms_to_idx_source = "generated"

            writer = VoxelH5Writer(
                outfile=tmp_path,
                t_bins=representation_channels,
                height=effective_output_height,
                width=effective_output_width,
                voxel_dtype=voxel_dtype,
                activity_mode=activity_mode,
                activity_grid_shape=activity_grid_shape,
                capacity_growth=writer_capacity_growth,
            )
            writer.h5f.attrs["representation"] = (
                "event_voxel_grid_1mpx" if representation == "voxel_grid" else "event_image_1mpx"
            )
            writer.h5f.attrs["representation_kind"] = str(representation)
            writer.h5f.attrs["input_height"] = int(resolved_input_height)
            writer.h5f.attrs["input_width"] = int(resolved_input_width)
            writer.h5f.attrs["requested_output_height"] = int(output_height)
            writer.h5f.attrs["requested_output_width"] = int(output_width)
            writer.h5f.attrs["height"] = int(effective_output_height)
            writer.h5f.attrs["width"] = int(effective_output_width)
            writer.h5f.attrs["downsample_factor"] = int(downsample_factor)
            writer.h5f.attrs["spatial_resize_mode"] = "nearest"
            writer.h5f.attrs["t_bins"] = int(activity_temporal_bins)
            writer.h5f.attrs["voxel_channels"] = int(representation_channels)
            writer.h5f.attrs["split_polarity"] = int(split_polarity if representation == "voxel_grid" else False)
            writer.h5f.attrs["polarity_channels"] = 2
            writer.h5f.attrs["accum_time_us"] = int(accum_time)
            writer.h5f.attrs["stride_time_us"] = int(stride_time)
            writer.h5f.attrs["normalize"] = int(normalize)
            writer.h5f.attrs["compression_level"] = int(compression_level)
            writer.h5f.attrs["trilinear_interpolation"] = int(use_trilinear)
            writer.h5f.attrs["event_image_percentile"] = float(event_image_percentile)
            writer.h5f.attrs["writer_capacity_growth"] = str(writer_capacity_growth)
            writer.h5f.attrs["rdcc_nbytes"] = int(rdcc_nbytes)
            writer.h5f.attrs["rdcc_nslots"] = int(rdcc_nslots)
            writer.h5f.attrs["rdcc_w0"] = float(rdcc_w0)
            writer.h5f.attrs["ms_to_idx_source"] = ms_to_idx_source
            writer.h5f.attrs["activity_mode"] = str(activity_mode)
            writer.h5f.attrs["activity_spatial_patch_size"] = int(activity_spatial_patch_size)
            writer.h5f.attrs["activity_temporal_patch_size"] = int(activity_temporal_patch_size)
            writer.h5f.attrs["has_companion_mp4"] = int(save_mp4)
            writer.h5f.attrs["companion_mp4_relpath"] = mp4_path.name if save_mp4 else ""
            writer.h5f.attrs["companion_mp4_fps"] = 0.0
            writer.h5f.attrs["companion_mp4_fps_source"] = ""

            if t_first is None or t_last_exclusive is None:
                writer.h5f.attrs["time_origin_us"] = -1
                writer.h5f.attrs["num_windows_planned"] = 0
                writer.close()
                writer = None
                os.replace(tmp_path, output_path)
                return

            time_origin_us = int(t_first) if start_time_us is None else int(start_time_us)
            window_start_us = max(int(t_first), time_origin_us)
            windows = _build_windows_from_start(
                start_us=window_start_us,
                end_exclusive_us=int(t_last_exclusive),
                accum_time_us=accum_time,
                stride_time_us=stride_time,
            )
            writer.h5f.attrs["time_origin_us"] = int(time_origin_us)
            writer.h5f.attrs["num_windows_planned"] = int(len(windows))

            voxelizer = None
            if representation == "voxel_grid":
                voxelizer = EventVoxelGrid(
                    input_size=(t_bins, effective_output_height, effective_output_width),
                    normalize=normalize,
                    separate_polarity=split_polarity,
                    trilinear_interpolation=use_trilinear,
                )
            if show_progress:
                pbar = tqdm.tqdm(total=len(windows), desc=input_path.name, leave=False)

            for window_index, start_us, end_us, anchor_us in windows:
                events = _extract_events_by_time(
                    filehandle=h5f,
                    start_us=start_us,
                    end_us=end_us,
                    ms_to_idx=ms_to_idx,
                )
                if representation == "voxel_grid":
                    assert voxelizer is not None
                    window_tensor = _events_to_voxel_numpy(
                        events=events,
                        voxelizer=voxelizer,
                        input_height=resolved_input_height,
                        input_width=resolved_input_width,
                        output_height=effective_output_height,
                        output_width=effective_output_width,
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
                        input_height=resolved_input_height,
                        input_width=resolved_input_width,
                        output_height=effective_output_height,
                        output_width=effective_output_width,
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
                        mp4_writer.write_rgb(frame_rgb, timestamp_us=anchor_us)
                window_tensor = window_tensor.astype(voxel_dtype, copy=False)
                writer.add_window(
                    voxel=window_tensor,
                    window_index=window_index,
                    t_start_us=start_us,
                    t_end_us=end_us,
                    rel_start_us=start_us - time_origin_us,
                    rel_end_us=end_us - time_origin_us,
                    anchor_timestamp_us=anchor_us,
                    anchor_rel_timestamp_us=anchor_us - time_origin_us,
                    event_count=len(events["t"]),
                    activity_score=activity_score,
                    active_pixel_ratio=active_pixel_ratio,
                    activity_grid=activity_grid,
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
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    normalize: bool,
    output_dtype: str,
    compression_level: int,
    use_trilinear: bool,
    representation: str,
    event_image_percentile: float,
    save_mp4: bool,
    mp4_fps: float | None,
    writer_capacity_growth: str,
    rdcc_nbytes: int,
    rdcc_nslots: int,
    rdcc_w0: float,
    tmp_suffix: str,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
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
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                downsample_factor=downsample_factor,
                t_bins=t_bins,
                split_polarity=split_polarity,
                accum_time=accum_time,
                stride_time=stride_time,
                start_time_us=start_time_us,
                normalize=normalize,
                output_dtype=output_dtype,
                compression_level=compression_level,
                use_trilinear=use_trilinear,
                representation=representation,
                event_image_percentile=event_image_percentile,
                save_mp4=save_mp4,
                mp4_fps=mp4_fps,
                writer_capacity_growth=writer_capacity_growth,
                rdcc_nbytes=rdcc_nbytes,
                rdcc_nslots=rdcc_nslots,
                rdcc_w0=rdcc_w0,
                activity_mode=activity_mode,
                activity_spatial_patch_size=activity_spatial_patch_size,
                activity_temporal_patch_size=activity_temporal_patch_size,
                show_progress=False,
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
    ok, err = _process_file_with_retry(
        input_path=input_path,
        output_path=output_path,
        input_height=job["input_height"],
        input_width=job["input_width"],
        output_height=job["output_height"],
        output_width=job["output_width"],
        downsample_factor=job["downsample_factor"],
        t_bins=job["t_bins"],
        split_polarity=job["split_polarity"],
        accum_time=job["accum_time"],
        stride_time=job["stride_time"],
        start_time_us=job["start_time_us"],
        normalize=job["normalize"],
        output_dtype=job["output_dtype"],
        compression_level=job["compression_level"],
        use_trilinear=job["use_trilinear"],
        representation=job["representation"],
        event_image_percentile=job["event_image_percentile"],
        save_mp4=job["save_mp4"],
        mp4_fps=job["mp4_fps"],
        writer_capacity_growth=job["writer_capacity_growth"],
        rdcc_nbytes=job["rdcc_nbytes"],
        rdcc_nslots=job["rdcc_nslots"],
        rdcc_w0=job["rdcc_w0"],
        tmp_suffix=job["tmp_suffix"],
        activity_mode=job["activity_mode"],
        activity_spatial_patch_size=job["activity_spatial_patch_size"],
        activity_temporal_patch_size=job["activity_temporal_patch_size"],
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


def _find_h5_files(dataset_root: Path, splits: list[str], recursive: bool) -> list[Path]:
    input_files: list[Path] = []
    for split in splits:
        if split in {"", "."}:
            split_dir = dataset_root
        else:
            split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] missing split directory: {split_dir}")
            continue

        files = split_dir.rglob("*.h5") if recursive else split_dir.glob("*.h5")
        input_files.extend(sorted([p for p in files if p.is_file()]))
    return input_files


def _build_output_path(
    input_path: Path,
    dataset_root: Path,
    output_root: Path | None,
    normalized_suffix: str,
    normalized_subdir: str | None,
    downsample_factor: int,
) -> Path:
    output_name = f"{input_path.stem}{normalized_suffix}"
    output_name = ensure_scale_tag_in_filename(output_name, downsample_factor=downsample_factor)
    if output_root is None:
        output_dir = input_path.parent
    else:
        relative_input = input_path.relative_to(dataset_root)
        output_dir = output_root / relative_input.parent

    if normalized_subdir is not None:
        output_dir = output_dir / normalized_subdir
    return output_dir / output_name


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
    output_suffix: str,
    output_subdir: str | None,
    overwrite: bool,
    output_root: Path | None,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    normalize: bool,
    output_dtype: str,
    compression_level: int,
    use_trilinear: bool,
    representation: str,
    event_image_percentile: float,
    save_mp4: bool,
    mp4_fps: float | None,
    writer_capacity_growth: str,
    rdcc_nbytes: int,
    rdcc_nslots: int,
    rdcc_w0: float,
    recursive: bool,
    tmp_suffix: str,
    num_processes: int,
    activity_mode: str,
    activity_spatial_patch_size: int,
    activity_temporal_patch_size: int,
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
) -> None:
    if int(num_processes) < 1:
        raise ValueError("num_processes must be >= 1")
    if representation not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported representation: {representation}")
    if int(compression_level) < 0 or int(compression_level) > 9:
        raise ValueError("compression_level must be in [0,9]")
    if writer_capacity_growth not in {"double", "exact"}:
        raise ValueError("writer_capacity_growth must be 'double' or 'exact'")
    if int(rdcc_nbytes) <= 0:
        raise ValueError("rdcc_nbytes must be > 0")
    if int(rdcc_nslots) <= 0:
        raise ValueError("rdcc_nslots must be > 0")
    if float(rdcc_w0) < 0.0 or float(rdcc_w0) > 1.0:
        raise ValueError("rdcc_w0 must be in [0,1]")
    if save_mp4 and representation != "event_image":
        raise ValueError("save_mp4 requires representation=event_image")
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")
    if int(activity_spatial_patch_size) <= 0 or int(activity_temporal_patch_size) <= 0:
        raise ValueError("activity patch sizes must be > 0")
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

    normalized_suffix = normalized_output_suffix(output_suffix)
    normalized_subdir = normalized_output_subdir(output_subdir)
    tagged_suffixes = {
        normalized_suffix,
        ensure_scale_tag_in_filename(f"dummy{normalized_suffix}", downsample_factor=1).removeprefix("dummy"),
        ensure_scale_tag_in_filename(f"dummy{normalized_suffix}", downsample_factor=2).removeprefix("dummy"),
    }
    input_files = _find_h5_files(dataset_root=dataset_root, splits=splits, recursive=recursive)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .h5 files found under {dataset_root} for splits={splits}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
    if split_output_root is not None:
        split_output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    num_done = 0
    num_skipped = 0
    num_failed = 0

    for input_path in tqdm.tqdm(input_files, desc="1mpx files"):
        if any(input_path.name.endswith(sfx) for sfx in tagged_suffixes):
            num_skipped += 1
            continue

        output_path = _build_output_path(
            input_path=input_path,
            dataset_root=dataset_root,
            output_root=output_root,
            normalized_suffix=normalized_suffix,
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

        jobs.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "downsample_factor": downsample_factor,
                "t_bins": t_bins,
                "split_polarity": bool(split_polarity),
                "accum_time": accum_time,
                "stride_time": stride_time,
                "start_time_us": start_time_us,
                "normalize": normalize,
                "output_dtype": output_dtype,
                "compression_level": int(compression_level),
                "use_trilinear": bool(use_trilinear),
                "representation": str(representation),
                "event_image_percentile": float(event_image_percentile),
                "save_mp4": bool(save_mp4),
                "mp4_fps": None if mp4_fps is None else float(mp4_fps),
                "writer_capacity_growth": str(writer_capacity_growth),
                "rdcc_nbytes": int(rdcc_nbytes),
                "rdcc_nslots": int(rdcc_nslots),
                "rdcc_w0": float(rdcc_w0),
                "tmp_suffix": tmp_suffix,
                "activity_mode": str(activity_mode),
                "activity_spatial_patch_size": int(activity_spatial_patch_size),
                "activity_temporal_patch_size": int(activity_temporal_patch_size),
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
            for input_name, success, err in tqdm.tqdm(iterator, total=len(jobs), desc="1mpx workers"):
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
                    desc="1mpx workers",
                ):
                    if success:
                        num_done += 1
                    else:
                        num_failed += 1
                        print(f"[FAILED] {input_name}: {err}")

    print(f"[SUMMARY] done={num_done}, skipped={num_skipped}, failed={num_failed}")
    if num_failed > 0:
        raise RuntimeError(f"{num_failed} files failed while processing {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build event representations from 1MPX-style events H5")
    parser.add_argument("--input_path", type=Path, help="Input events .h5")
    parser.add_argument("--output_path", type=Path, help="Output voxel .h5")

    parser.add_argument("--dataset_root", type=Path, help="Dataset root with split directories")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "val"],
        help="Splits for --dataset_root mode. Use '.' to scan dataset_root directly.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_voxels.h5",
        help="Output suffix for --dataset_root mode (e.g. _voxels.h5). Scale tag (_1x/_2x) is auto-added.",
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
        help="Optional output root dir for --dataset_root mode (preserves relative split paths)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--recursive", action="store_true", help="Recursively search .h5 under split dirs")
    parser.add_argument(
        "--tmp_suffix",
        type=str,
        default=".tmp",
        help="Temporary suffix used while writing (renamed on success)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Parallel workers for --dataset_root mode (spawn).",
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
    parser.add_argument(
        "--activity_spatial_patch_size",
        type=int,
        default=16,
        help="Spatial patch size used when aggregating activity metadata.",
    )
    parser.add_argument(
        "--activity_temporal_patch_size",
        type=int,
        default=2,
        help="Temporal patch size used for full activity metadata.",
    )

    parser.add_argument("--input_height", type=int, default=720, help="Input event height (default: 720).")
    parser.add_argument("--input_width", type=int, default=1280, help="Input event width (default: 1280).")
    parser.add_argument("--output_height", type=int, default=720, help="Output voxel height (default: 720).")
    parser.add_argument("--output_width", type=int, default=1280, help="Output voxel width (default: 1280).")
    parser.add_argument(
        "--downsample_factor",
        type=int,
        choices=[1, 2],
        default=2,
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
        help="Optional fixed time origin in us. Default uses the first event timestamp in each file.",
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
        "--compression_level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help="Compression level in [0,9] for Blosc(gzip fallback). Higher is smaller but slower.",
    )
    parser.add_argument(
        "--use_trilinear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use trilinear interpolation in voxelization (voxel_grid only).",
    )
    parser.add_argument(
        "--writer_capacity_growth",
        choices=["double", "exact"],
        default="double",
        help="Dataset capacity growth policy while writing windows.",
    )
    parser.add_argument(
        "--rdcc_nbytes",
        type=int,
        default=256 * 1024 * 1024,
        help="HDF5 raw chunk cache bytes for input reads.",
    )
    parser.add_argument(
        "--rdcc_nslots",
        type=int,
        default=1000003,
        help="HDF5 raw chunk cache slots for input reads (use a large prime).",
    )
    parser.add_argument(
        "--rdcc_w0",
        type=float,
        default=0.25,
        help="HDF5 raw chunk cache preemption policy [0,1].",
    )
    args = parser.parse_args()

    stride_time = args.accum_time if args.stride_time is None else args.stride_time

    is_single_mode = args.input_path is not None or args.output_path is not None
    is_root_mode = args.dataset_root is not None
    if is_single_mode and is_root_mode:
        parser.error("Use either single-file mode (--input_path/--output_path) or root mode (--dataset_root), not both.")

    if is_root_mode:
        process_dataset_root(
            dataset_root=args.dataset_root,
            splits=args.splits,
            output_suffix=args.output_suffix,
            output_subdir=args.output_subdir,
            overwrite=args.overwrite,
            output_root=args.output_root,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            split_polarity=args.split_polarity,
            accum_time=args.accum_time,
            stride_time=stride_time,
            start_time_us=args.start_time_us,
            normalize=args.normalize,
            output_dtype=args.output_dtype,
            compression_level=args.compression_level,
            use_trilinear=args.use_trilinear,
            representation=args.representation,
            event_image_percentile=args.event_image_percentile,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
            writer_capacity_growth=args.writer_capacity_growth,
            rdcc_nbytes=args.rdcc_nbytes,
            rdcc_nslots=args.rdcc_nslots,
            rdcc_w0=args.rdcc_w0,
            recursive=args.recursive,
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
            activity_mode=args.activity_mode,
            activity_spatial_patch_size=args.activity_spatial_patch_size,
            activity_temporal_patch_size=args.activity_temporal_patch_size,
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

        process_single_file(
            input_path=args.input_path,
            output_path=args.output_path,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            split_polarity=args.split_polarity,
            accum_time=args.accum_time,
            stride_time=stride_time,
            start_time_us=args.start_time_us,
            normalize=args.normalize,
            output_dtype=args.output_dtype,
            compression_level=args.compression_level,
            use_trilinear=args.use_trilinear,
            representation=args.representation,
            event_image_percentile=args.event_image_percentile,
            save_mp4=args.save_mp4,
            mp4_fps=args.mp4_fps,
            writer_capacity_growth=args.writer_capacity_growth,
            rdcc_nbytes=args.rdcc_nbytes,
            rdcc_nslots=args.rdcc_nslots,
            rdcc_w0=args.rdcc_w0,
            activity_mode=args.activity_mode,
            activity_spatial_patch_size=args.activity_spatial_patch_size,
            activity_temporal_patch_size=args.activity_temporal_patch_size,
            show_progress=True,
            tmp_suffix=args.tmp_suffix,
        )
