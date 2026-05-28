from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
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


H5_COMPRESSION_FLAGS = get_h5_compression_flags()
ACTIVITY_MODES = {"full", "light"}
REPRESENTATION_MODES = {"voxel_grid", "event_image"}

EVENTS_SUBDIR = Path("events/data")
SEMANTIC_SUBDIR = Path("semantic/data")
DEPTH_SUBDIR = Path("depth/data")


def _empty_events() -> dict[str, np.ndarray]:
    return {
        "x": np.empty((0,), dtype=np.uint16),
        "y": np.empty((0,), dtype=np.uint16),
        "p": np.empty((0,), dtype=np.int16),
        "t": np.empty((0,), dtype=np.int64),
    }


def _parse_frame_index(path: Path) -> int | None:
    matches = re.findall(r"(\d+)", path.stem)
    if len(matches) == 0:
        return None
    return int(matches[-1])


def _load_timestamp_txt(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.loadtxt(str(path))
    arr = np.atleast_1d(arr).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.int64)
    return arr.astype(np.int64, copy=False)


def _collect_event_files(sequence_dir: Path) -> list[Path]:
    events_dir = sequence_dir / EVENTS_SUBDIR
    if not events_dir.exists() or not events_dir.is_dir():
        return []

    files = sorted([p for p in events_dir.glob("*.npz") if p.is_file()])
    # Prefer canonical EventScape naming if available.
    canonical = [p for p in files if p.name.endswith("_events.npz")]
    if len(canonical) > 0:
        files = canonical

    files.sort(key=lambda p: (_parse_frame_index(p) is None, _parse_frame_index(p) or -1, p.name))
    return files


def _collect_indexed_files(directory: Path, pattern: str) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    if not directory.exists() or not directory.is_dir():
        return mapping

    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        idx = _parse_frame_index(path)
        if idx is None:
            continue
        mapping[idx] = path
    return mapping


def _squeeze_label_to_hw(arr: np.ndarray, *, name: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"unsupported {name} label shape={arr.shape}; expected HxW or singleton-channel variant")


def _load_semantic_png(path: Path) -> np.ndarray:
    with Image.open(str(path)) as img:
        arr = np.asarray(img)
    return _squeeze_label_to_hw(arr, name="EventScape semantic")


def _load_depth_npy(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(str(path), allow_pickle=False))
    return _squeeze_label_to_hw(arr, name="EventScape depth")


def _infer_embedded_label_spec(
    files_by_idx: dict[int, Path],
    *,
    loader,
) -> tuple[tuple[int, int], np.dtype] | None:
    for path in files_by_idx.values():
        arr = np.asarray(loader(path))
        shape = tuple(int(v) for v in arr.shape)
        if len(shape) != 2:
            raise ValueError(f"embedded label must be HxW, got shape={shape} from {path}")
        return shape, arr.dtype
    return None


def _first_existing_array(npz_data, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in npz_data:
            return np.asarray(npz_data[key])
    return None


def _extract_from_structured(arr: np.ndarray, names: tuple[str, ...]) -> np.ndarray | None:
    if arr.dtype.names is None:
        return None
    for name in names:
        if name in arr.dtype.names:
            return np.asarray(arr[name])
    return None


def _extract_events_from_2d(arr: np.ndarray) -> dict[str, np.ndarray] | None:
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None

    # Common fallback convention: [x, y, t, p]
    x = np.asarray(arr[:, 0])
    y = np.asarray(arr[:, 1])
    t = np.asarray(arr[:, 2])
    p = np.asarray(arr[:, 3])
    return {"x": x, "y": y, "t": t, "p": p}


def load_eventscape_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=False) as npz_data:
        x = _first_existing_array(npz_data, ("x", "xs", "x_coord", "x_coords"))
        y = _first_existing_array(npz_data, ("y", "ys", "y_coord", "y_coords"))
        t = _first_existing_array(npz_data, ("t", "ts", "timestamp", "timestamps"))
        p = _first_existing_array(npz_data, ("p", "pol", "polarity", "polarities"))

        if x is None or y is None or t is None or p is None:
            for key in ("events", "arr_0"):
                if key not in npz_data:
                    continue
                arr = np.asarray(npz_data[key])
                structured_x = _extract_from_structured(arr, ("x", "xs"))
                structured_y = _extract_from_structured(arr, ("y", "ys"))
                structured_t = _extract_from_structured(arr, ("t", "ts", "timestamp", "timestamps"))
                structured_p = _extract_from_structured(arr, ("p", "pol", "polarity"))

                if structured_x is not None:
                    x = structured_x
                if structured_y is not None:
                    y = structured_y
                if structured_t is not None:
                    t = structured_t
                if structured_p is not None:
                    p = structured_p

                if x is not None and y is not None and t is not None and p is not None:
                    break

                extracted = _extract_events_from_2d(arr)
                if extracted is not None:
                    x = extracted["x"]
                    y = extracted["y"]
                    t = extracted["t"]
                    p = extracted["p"]
                    break

        if x is None or y is None or t is None or p is None:
            raise KeyError(
                f"could not resolve event arrays from npz: {npz_path}. "
                f"available keys={sorted(npz_data.files)}"
            )

        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        t = np.asarray(t).reshape(-1)
        p = np.asarray(p).reshape(-1)
        if not (len(x) == len(y) == len(t) == len(p)):
            raise ValueError(
                f"inconsistent event array lengths in {npz_path}: "
                f"x={len(x)}, y={len(y)}, t={len(t)}, p={len(p)}"
            )

        return {
            "x": x.astype(np.uint16, copy=False),
            "y": y.astype(np.uint16, copy=False),
            "t": t.astype(np.int64, copy=False),
            "p": p.astype(np.int16, copy=False),
        }


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
    t_src = events["t"].astype(np.int64, copy=False)

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
        x_out = np.floor(x_src / float(fx)).astype(np.float32, copy=False)
        y_out = np.floor(y_src / float(fy)).astype(np.float32, copy=False)
    else:
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

    p_bin = normalize_polarity_to_binary(p_src, dtype=np.float32)
    t_out = t_src[valid_out].astype(np.int64, copy=False)
    x_out = x_out[valid_out].astype(np.float32, copy=False)
    y_out = y_out[valid_out].astype(np.float32, copy=False)
    p_out = p_bin[valid_out].astype(np.float32, copy=False)

    if t_out.size > 1 and np.any(t_out[1:] < t_out[:-1]):
        order = np.argsort(t_out, kind="stable")
        x_out = x_out[order]
        y_out = y_out[order]
        p_out = p_out[order]
        t_out = t_out[order]

    return {
        "x": x_out,
        "y": y_out,
        "p": p_out,
        "t": t_out,
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


class VoxelH5Writer:
    def __init__(
        self,
        outfile: Path,
        t_bins: int,
        height: int,
        width: int,
        voxel_dtype: np.dtype,
        activity_grid_shape: tuple[int, ...],
        with_embedded_semantics: bool = False,
        embedded_semantics_shape: tuple[int, int] = (1, 1),
        embedded_semantics_dtype: np.dtype = np.uint8,
        with_embedded_depth: bool = False,
        embedded_depth_shape: tuple[int, int] = (1, 1),
        embedded_depth_dtype: np.dtype = np.float32,
        initial_capacity: int = 256,
    ):
        if outfile.exists():
            raise FileExistsError(f"output already exists: {outfile}")

        self.h5f = h5py.File(str(outfile), "a")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self._capacity = int(initial_capacity)
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
            "event_frame_index",
            "event_file_relpath",
            "embedded_semantics",
            "semantic_available",
            "semantic_frame_index",
            "semantic_timestamp_us",
            "semantic_relpath",
            "embedded_depth",
            "depth_available",
            "depth_frame_index",
            "depth_timestamp_us",
            "depth_relpath",
            "window_activity_score",
            "window_active_pixel_ratio",
            "window_activity_grid",
        )

        voxel_chunks = (1, t_bins, min(height, 64), min(width, 64))
        scalar_chunks = (min(self._capacity, 4096),)
        string_dtype = h5py.string_dtype(encoding="utf-8")

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
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_t_end_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
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
            "event_frame_index",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "event_file_relpath",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=scalar_chunks,
        )
        if with_embedded_semantics:
            sem_h, sem_w = embedded_semantics_shape
            self.h5f.create_dataset(
                "embedded_semantics",
                shape=(self._capacity, int(sem_h), int(sem_w)),
                maxshape=(None, int(sem_h), int(sem_w)),
                dtype=embedded_semantics_dtype,
                chunks=(1, min(int(sem_h), 256), min(int(sem_w), 256)),
                **H5_COMPRESSION_FLAGS,
            )
        else:
            self._datasets = tuple(d for d in self._datasets if d != "embedded_semantics")
        self.h5f.create_dataset(
            "semantic_available",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u1",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "semantic_frame_index",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "semantic_timestamp_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "semantic_relpath",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=scalar_chunks,
        )
        if with_embedded_depth:
            depth_h, depth_w = embedded_depth_shape
            self.h5f.create_dataset(
                "embedded_depth",
                shape=(self._capacity, int(depth_h), int(depth_w)),
                maxshape=(None, int(depth_h), int(depth_w)),
                dtype=embedded_depth_dtype,
                chunks=(1, min(int(depth_h), 256), min(int(depth_w), 256)),
                **H5_COMPRESSION_FLAGS,
            )
        else:
            self._datasets = tuple(d for d in self._datasets if d != "embedded_depth")
        self.h5f.create_dataset(
            "depth_available",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u1",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "depth_frame_index",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "depth_timestamp_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "depth_relpath",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype=string_dtype,
            chunks=scalar_chunks,
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
        new_capacity = self._capacity
        while new_capacity < needed:
            new_capacity *= 2
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
        event_frame_index: int,
        event_file_relpath: str,
        embedded_semantics: np.ndarray | None,
        semantic_available: int,
        semantic_frame_index: int,
        semantic_timestamp_us: int,
        semantic_relpath: str,
        embedded_depth: np.ndarray | None,
        depth_available: int,
        depth_frame_index: int,
        depth_timestamp_us: int,
        depth_relpath: str,
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
        self.h5f["event_frame_index"][idx] = int(event_frame_index)
        self.h5f["event_file_relpath"][idx] = event_file_relpath
        if "embedded_semantics" in self.h5f:
            if embedded_semantics is None:
                raise ValueError("embedded_semantics is required when embedded_semantics dataset is present")
            self.h5f["embedded_semantics"][idx] = embedded_semantics
        self.h5f["semantic_available"][idx] = int(semantic_available)
        self.h5f["semantic_frame_index"][idx] = int(semantic_frame_index)
        self.h5f["semantic_timestamp_us"][idx] = int(semantic_timestamp_us)
        self.h5f["semantic_relpath"][idx] = semantic_relpath
        if "embedded_depth" in self.h5f:
            if embedded_depth is None:
                raise ValueError("embedded_depth is required when embedded_depth dataset is present")
            self.h5f["embedded_depth"][idx] = embedded_depth
        self.h5f["depth_available"][idx] = int(depth_available)
        self.h5f["depth_frame_index"][idx] = int(depth_frame_index)
        self.h5f["depth_timestamp_us"][idx] = int(depth_timestamp_us)
        self.h5f["depth_relpath"][idx] = depth_relpath
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


def _resolve_semantic_meta(
    frame_idx: int | None,
    sequence_dir: Path,
    semantic_by_idx: dict[int, Path],
    semantic_timestamps: np.ndarray | None,
) -> tuple[int, int, int, str]:
    if frame_idx is None:
        return 0, -1, -1, ""

    sem_path = semantic_by_idx.get(int(frame_idx))
    if sem_path is None:
        return 0, int(frame_idx), -1, ""

    sem_ts = -1
    if semantic_timestamps is not None and 0 <= int(frame_idx) < len(semantic_timestamps):
        sem_ts = int(semantic_timestamps[int(frame_idx)])
    return 1, int(frame_idx), sem_ts, str(sem_path.relative_to(sequence_dir))


def _resolve_depth_meta(
    frame_idx: int | None,
    sequence_dir: Path,
    depth_by_idx: dict[int, Path],
    depth_timestamps: np.ndarray | None,
) -> tuple[int, int, int, str]:
    if frame_idx is None:
        return 0, -1, -1, ""

    depth_path = depth_by_idx.get(int(frame_idx))
    if depth_path is None:
        return 0, int(frame_idx), -1, ""

    depth_ts = -1
    if depth_timestamps is not None and 0 <= int(frame_idx) < len(depth_timestamps):
        depth_ts = int(depth_timestamps[int(frame_idx)])
    return 1, int(frame_idx), depth_ts, str(depth_path.relative_to(sequence_dir))


def _build_output_path(
    sequence_dir: Path,
    dataset_root: Path,
    output_root: Path | None,
    normalized_suffix: str,
    normalized_subdir: str | None,
    downsample_factor: int,
) -> Path:
    output_name = f"{sequence_dir.name}{normalized_suffix}"
    output_name = ensure_scale_tag_in_filename(output_name, downsample_factor=downsample_factor)

    if output_root is None:
        output_dir = sequence_dir
    else:
        rel_sequence = sequence_dir.relative_to(dataset_root)
        output_dir = output_root / rel_sequence

    if normalized_subdir is not None:
        output_dir = output_dir / normalized_subdir
    return output_dir / output_name


def process_sequence(
    sequence_dir: Path,
    output_path: Path,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
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
    show_progress: bool,
    tmp_suffix: str,
) -> None:
    if input_height <= 0 or input_width <= 0:
        raise ValueError("input_height/input_width must be > 0")
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_height/output_width must be > 0")
    if int(downsample_factor) not in (1, 2):
        raise ValueError("downsample_factor must be 1 or 2")
    if t_bins <= 0:
        raise ValueError("t_bins must be > 0")
    if representation not in REPRESENTATION_MODES:
        raise ValueError(f"unsupported representation: {representation}")
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")

    event_files = _collect_event_files(sequence_dir)
    if len(event_files) == 0:
        raise FileNotFoundError(f"no event npz files found in {sequence_dir / EVENTS_SUBDIR}")

    semantic_data_dir = sequence_dir / SEMANTIC_SUBDIR
    depth_data_dir = sequence_dir / DEPTH_SUBDIR
    semantic_by_idx = _collect_indexed_files(semantic_data_dir, "*.png")
    depth_by_idx = _collect_indexed_files(depth_data_dir, "*.npy")
    semantic_timestamps = _load_timestamp_txt(semantic_data_dir / "timestamps.txt")
    depth_timestamps = _load_timestamp_txt(depth_data_dir / "timestamps.txt")
    semantic_spec = _infer_embedded_label_spec(semantic_by_idx, loader=_load_semantic_png)
    depth_spec = _infer_embedded_label_spec(depth_by_idx, loader=_load_depth_npy)

    effective_output_height, effective_output_width = _resolve_output_resolution(
        input_height=input_height,
        input_width=input_width,
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
    cleanup_tmp_file(tmp_path=tmp_path, context=f"start processing {sequence_dir}", strict=True)
    mp4_path = output_path.with_suffix(".mp4")
    tmp_mp4_path = tmp_media_output_path(output_path=mp4_path, tmp_suffix=tmp_suffix)
    if save_mp4 and representation != "event_image":
        raise ValueError("save_mp4 is only supported when representation=event_image")
    if save_mp4:
        cleanup_tmp_file(tmp_path=tmp_mp4_path, context=f"start MP4 processing {sequence_dir}", strict=True)

    writer = None
    mp4_writer = None
    pbar = None
    try:
        writer = VoxelH5Writer(
            outfile=tmp_path,
            t_bins=representation_channels,
            height=effective_output_height,
            width=effective_output_width,
            voxel_dtype=voxel_dtype,
            activity_grid_shape=activity_grid_shape,
            with_embedded_semantics=semantic_spec is not None,
            embedded_semantics_shape=((1, 1) if semantic_spec is None else semantic_spec[0]),
            embedded_semantics_dtype=(np.uint8 if semantic_spec is None else semantic_spec[1]),
            with_embedded_depth=depth_spec is not None,
            embedded_depth_shape=((1, 1) if depth_spec is None else depth_spec[0]),
            embedded_depth_dtype=(np.float32 if depth_spec is None else depth_spec[1]),
            initial_capacity=max(256, len(event_files)),
        )
        writer.h5f.attrs["representation"] = (
            "event_voxel_grid_eventscape" if representation == "voxel_grid" else "event_image_eventscape"
        )
        writer.h5f.attrs["representation_kind"] = str(representation)
        writer.h5f.attrs["input_height"] = int(input_height)
        writer.h5f.attrs["input_width"] = int(input_width)
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
        writer.h5f.attrs["window_mode"] = "event_file"
        writer.h5f.attrs["normalize"] = int(normalize)
        writer.h5f.attrs["trilinear_interpolation"] = int(use_trilinear)
        writer.h5f.attrs["event_image_percentile"] = float(event_image_percentile)
        writer.h5f.attrs["ms_to_idx_source"] = "not_applicable_npz"
        writer.h5f.attrs["num_event_files"] = int(len(event_files))
        writer.h5f.attrs["num_semantic_files"] = int(len(semantic_by_idx))
        writer.h5f.attrs["num_depth_files"] = int(len(depth_by_idx))
        writer.h5f.attrs["has_semantic_timestamps"] = int(semantic_timestamps is not None)
        writer.h5f.attrs["has_depth_timestamps"] = int(depth_timestamps is not None)
        writer.h5f.attrs["embedded_semantic_dataset"] = "embedded_semantics" if semantic_spec is not None else ""
        writer.h5f.attrs["embedded_depth_dataset"] = "embedded_depth" if depth_spec is not None else ""
        writer.h5f.attrs["activity_mode"] = str(activity_mode)
        writer.h5f.attrs["activity_spatial_patch_size"] = int(activity_spatial_patch_size)
        writer.h5f.attrs["activity_temporal_patch_size"] = int(activity_temporal_patch_size)
        writer.h5f.attrs["has_companion_mp4"] = int(save_mp4)
        writer.h5f.attrs["companion_mp4_relpath"] = mp4_path.name if save_mp4 else ""
        writer.h5f.attrs["companion_mp4_fps"] = 0.0
        writer.h5f.attrs["companion_mp4_fps_source"] = ""

        voxelizer = None
        if representation == "voxel_grid":
            voxelizer = EventVoxelGrid(
                input_size=(t_bins, effective_output_height, effective_output_width),
                normalize=normalize,
                separate_polarity=split_polarity,
                trilinear_interpolation=use_trilinear,
            )
        if show_progress:
            pbar = tqdm.tqdm(total=len(event_files), desc=sequence_dir.name, leave=False)

        time_origin_us: int | None = None
        for window_index, event_file in enumerate(event_files):
            frame_idx = _parse_frame_index(event_file)
            if frame_idx is None:
                frame_idx = int(window_index)

            events = load_eventscape_npz(event_file)
            event_count = len(events["t"])
            if event_count > 0:
                t_start_us = int(events["t"][0])
                t_end_us = int(events["t"][-1]) + 1
                if t_end_us <= t_start_us:
                    t_end_us = t_start_us + 1
                anchor_us = t_start_us + (t_end_us - t_start_us) // 2
            else:
                t_start_us = 0
                t_end_us = 0
                anchor_us = 0

            if time_origin_us is None:
                if event_count > 0:
                    time_origin_us = int(t_start_us)
                else:
                    time_origin_us = 0

            semantic_available, semantic_frame_idx, semantic_timestamp_us, semantic_relpath = _resolve_semantic_meta(
                frame_idx=frame_idx,
                sequence_dir=sequence_dir,
                semantic_by_idx=semantic_by_idx,
                semantic_timestamps=semantic_timestamps,
            )
            depth_available, depth_frame_idx, depth_timestamp_us, depth_relpath = _resolve_depth_meta(
                frame_idx=frame_idx,
                sequence_dir=sequence_dir,
                depth_by_idx=depth_by_idx,
                depth_timestamps=depth_timestamps,
            )
            embedded_semantics = None
            if semantic_spec is not None:
                semantic_shape, semantic_dtype = semantic_spec
                if semantic_available == 1 and semantic_frame_idx >= 0:
                    semantic_path = semantic_by_idx.get(int(semantic_frame_idx))
                    if semantic_path is None:
                        raise FileNotFoundError(
                            f"missing EventScape semantic file for frame {semantic_frame_idx} in {sequence_dir}"
                        )
                    embedded_semantics = np.asarray(_load_semantic_png(semantic_path))
                    if tuple(int(v) for v in embedded_semantics.shape) != tuple(int(v) for v in semantic_shape):
                        raise ValueError(
                            "Inconsistent EventScape semantic shape: "
                            f"expected {semantic_shape}, got {embedded_semantics.shape} for {semantic_path}"
                        )
                    embedded_semantics = embedded_semantics.astype(semantic_dtype, copy=False)
                else:
                    embedded_semantics = np.full(semantic_shape, fill_value=255, dtype=semantic_dtype)
            embedded_depth = None
            if depth_spec is not None:
                depth_shape, depth_dtype = depth_spec
                if depth_available == 1 and depth_frame_idx >= 0:
                    depth_path = depth_by_idx.get(int(depth_frame_idx))
                    if depth_path is None:
                        raise FileNotFoundError(
                            f"missing EventScape depth file for frame {depth_frame_idx} in {sequence_dir}"
                        )
                    embedded_depth = np.asarray(_load_depth_npy(depth_path))
                    if tuple(int(v) for v in embedded_depth.shape) != tuple(int(v) for v in depth_shape):
                        raise ValueError(
                            "Inconsistent EventScape depth shape: "
                            f"expected {depth_shape}, got {embedded_depth.shape} for {depth_path}"
                        )
                    embedded_depth = embedded_depth.astype(depth_dtype, copy=False)
                else:
                    if np.issubdtype(depth_dtype, np.floating):
                        embedded_depth = np.full(depth_shape, fill_value=np.nan, dtype=depth_dtype)
                    else:
                        embedded_depth = np.zeros(depth_shape, dtype=depth_dtype)
            if anchor_us == 0 and semantic_available == 1 and semantic_timestamp_us >= 0:
                anchor_us = int(semantic_timestamp_us)
                t_start_us = int(semantic_timestamp_us)
                t_end_us = int(semantic_timestamp_us)

            if representation == "voxel_grid":
                assert voxelizer is not None
                window_tensor = _events_to_voxel_numpy(
                    events=events,
                    voxelizer=voxelizer,
                    input_height=input_height,
                    input_width=input_width,
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
                    input_height=input_height,
                    input_width=input_width,
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
                t_start_us=t_start_us,
                t_end_us=t_end_us,
                rel_start_us=t_start_us - int(time_origin_us),
                rel_end_us=t_end_us - int(time_origin_us),
                anchor_timestamp_us=anchor_us,
                anchor_rel_timestamp_us=anchor_us - int(time_origin_us),
                event_count=event_count,
                event_frame_index=frame_idx,
                event_file_relpath=str(event_file.relative_to(sequence_dir)),
                embedded_semantics=embedded_semantics,
                semantic_available=semantic_available,
                semantic_frame_index=semantic_frame_idx,
                semantic_timestamp_us=semantic_timestamp_us,
                semantic_relpath=semantic_relpath,
                embedded_depth=embedded_depth,
                depth_available=depth_available,
                depth_frame_index=depth_frame_idx,
                depth_timestamp_us=depth_timestamp_us,
                depth_relpath=depth_relpath,
                activity_score=activity_score,
                active_pixel_ratio=active_pixel_ratio,
                activity_grid=activity_grid,
            )
            if pbar is not None:
                pbar.update(1)

        writer.h5f.attrs["time_origin_us"] = int(time_origin_us if time_origin_us is not None else 0)
        writer.h5f.attrs["num_windows_planned"] = int(len(event_files))
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
        cleanup_tmp_file(tmp_path=tmp_path, context=f"exception cleanup for {sequence_dir}", strict=False)
        if save_mp4:
            cleanup_tmp_file(tmp_path=tmp_mp4_path, context=f"exception MP4 cleanup for {sequence_dir}", strict=False)
        raise
    finally:
        if pbar is not None:
            pbar.close()


def _process_sequence_with_retry(
    sequence_dir: Path,
    output_path: Path,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
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
    tmp_suffix: str,
) -> tuple[bool, str | None]:
    stale_tmp_path = tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    if not cleanup_tmp_file(tmp_path=stale_tmp_path, context=f"resume prep for {sequence_dir}", strict=False):
        return False, f"could not remove stale tmp file: {stale_tmp_path}"

    for attempt in (1, 2):
        try:
            process_sequence(
                sequence_dir=sequence_dir,
                output_path=output_path,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                downsample_factor=downsample_factor,
                t_bins=t_bins,
                split_polarity=split_polarity,
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
                show_progress=False,
                tmp_suffix=tmp_suffix,
            )
            return True, None
        except Exception as exc:
            if attempt == 1:
                cleanup_ok = cleanup_tmp_file(
                    tmp_path=stale_tmp_path,
                    context=f"retry prep for {sequence_dir}",
                    strict=False,
                )
                if not cleanup_ok:
                    return False, f"retry cleanup failed for {stale_tmp_path}: {exc}"
                continue
            return False, str(exc)

    return False, "unknown failure"


def _worker_process_sequence(job: dict) -> tuple[str, bool, str | None]:
    sequence_dir = Path(job["sequence_dir"])
    output_path = Path(job["output_path"])
    ok, err = _process_sequence_with_retry(
        sequence_dir=sequence_dir,
        output_path=output_path,
        input_height=job["input_height"],
        input_width=job["input_width"],
        output_height=job["output_height"],
        output_width=job["output_width"],
        downsample_factor=job["downsample_factor"],
        t_bins=job["t_bins"],
        split_polarity=job["split_polarity"],
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
        tmp_suffix=job["tmp_suffix"],
    )
    return str(sequence_dir), ok, err


def find_eventscape_sequences(dataset_root: Path, splits: list[str] | None = None) -> list[Path]:
    sequences: list[Path] = []
    seen: set[Path] = set()

    search_roots: list[Path] = []
    if splits is None or len(splits) == 0:
        search_roots = [dataset_root]
    else:
        for split in splits:
            split_name = str(split).strip()
            if split_name in {"", "."}:
                base_dir = dataset_root
            else:
                base_dir = dataset_root / split_name
            if not base_dir.exists():
                print(f"[WARN] missing split directory: {base_dir}")
                continue
            search_roots.append(base_dir)

    for root in search_roots:
        for events_data_dir in sorted(root.rglob(str(EVENTS_SUBDIR))):
            if not events_data_dir.is_dir():
                continue
            if not any(events_data_dir.glob("*.npz")):
                continue
            sequence_dir = events_data_dir.parent.parent
            resolved = sequence_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            sequences.append(sequence_dir)

    sequences.sort(key=lambda p: str(p))
    return sequences


def process_dataset_root(
    dataset_root: Path,
    splits: list[str] | None,
    output_suffix: str,
    output_subdir: str | None,
    overwrite: bool,
    output_root: Path | None,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    split_polarity: bool,
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
    tmp_suffix: str,
    num_processes: int,
) -> None:
    if int(num_processes) < 1:
        raise ValueError("num_processes must be >= 1")
    if save_mp4 and representation != "event_image":
        raise ValueError("save_mp4 requires representation=event_image")
    if activity_mode not in ACTIVITY_MODES:
        raise ValueError(f"unsupported activity_mode: {activity_mode}")

    normalized_suffix = normalized_output_suffix(output_suffix)
    normalized_subdir = normalized_output_subdir(output_subdir)
    sequence_dirs = find_eventscape_sequences(dataset_root=dataset_root, splits=splits)
    if len(sequence_dirs) == 0:
        raise FileNotFoundError(f"no EventScape sequences found under {dataset_root}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    num_done = 0
    num_skipped = 0
    num_failed = 0

    for sequence_dir in tqdm.tqdm(sequence_dirs, desc="EventScape sequences"):
        output_path = _build_output_path(
            sequence_dir=sequence_dir,
            dataset_root=dataset_root,
            output_root=output_root,
            normalized_suffix=normalized_suffix,
            normalized_subdir=normalized_subdir,
            downsample_factor=downsample_factor,
        )
        if output_path.exists():
            if overwrite:
                output_path.unlink()
            else:
                num_skipped += 1
                continue

        jobs.append(
            {
                "sequence_dir": str(sequence_dir),
                "output_path": str(output_path),
                "input_height": int(input_height),
                "input_width": int(input_width),
                "output_height": int(output_height),
                "output_width": int(output_width),
                "downsample_factor": int(downsample_factor),
                "t_bins": int(t_bins),
                "split_polarity": bool(split_polarity),
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
                "tmp_suffix": tmp_suffix,
            }
        )

    if len(jobs) > 0:
        if int(num_processes) == 1:
            iterator = (_worker_process_sequence(job) for job in jobs)
            for sequence_name, success, err in tqdm.tqdm(iterator, total=len(jobs), desc="EventScape workers"):
                if success:
                    num_done += 1
                else:
                    num_failed += 1
                    print(f"[FAILED] {sequence_name}: {err}")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(num_processes)) as pool:
                for sequence_name, success, err in tqdm.tqdm(
                    pool.imap_unordered(_worker_process_sequence, jobs),
                    total=len(jobs),
                    desc="EventScape workers",
                ):
                    if success:
                        num_done += 1
                    else:
                        num_failed += 1
                        print(f"[FAILED] {sequence_name}: {err}")

    print(f"[SUMMARY] done={num_done}, skipped={num_skipped}, failed={num_failed}")
    if num_failed > 0:
        raise RuntimeError(f"{num_failed} sequences failed while processing {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build event voxel representations from EventScape event npz sequences")
    parser.add_argument("--input_path", type=Path, help="Input sequence directory (contains events/data)")
    parser.add_argument("--output_path", type=Path, help="Output voxel .h5 for single-sequence mode")

    parser.add_argument("--dataset_root", type=Path, help="EventScape root (e.g., contains Town01/Town05/...)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=(
            "Optional split subdirectories under dataset_root "
            "(e.g. Town01-03_train Town05_val Town05_test). "
            "Use '.' for dataset_root itself."
        ),
    )
    parser.add_argument(
        "--activity_mode",
        choices=["full", "light"],
        default="full",
        help="Activity metadata layout saved per window.",
    )
    parser.add_argument("--activity_spatial_patch_size", type=int, default=16, help="Spatial patch size for activity metadata.")
    parser.add_argument("--activity_temporal_patch_size", type=int, default=2, help="Temporal patch size for full activity metadata.")
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
        help="Optional output root dir for --dataset_root mode (preserves relative sequence paths)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
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

    parser.add_argument("--input_height", type=int, default=256, help="Input event height (default: 256).")
    parser.add_argument("--input_width", type=int, default=512, help="Input event width (default: 512).")
    parser.add_argument("--output_height", type=int, default=256, help="Output voxel height (default: 256).")
    parser.add_argument("--output_width", type=int, default=512, help="Output voxel width (default: 512).")
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
    args = parser.parse_args()

    is_single_mode = args.input_path is not None or args.output_path is not None
    is_root_mode = args.dataset_root is not None
    if is_single_mode and is_root_mode:
        parser.error("Use either single mode (--input_path/--output_path) or root mode (--dataset_root), not both.")

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
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
        )
    else:
        if args.input_path is None or args.output_path is None:
            parser.error("Single mode requires both --input_path and --output_path.")

        process_sequence(
            sequence_dir=args.input_path,
            output_path=args.output_path,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            split_polarity=args.split_polarity,
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
            show_progress=True,
            tmp_suffix=args.tmp_suffix,
        )
