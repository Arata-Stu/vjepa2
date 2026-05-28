from __future__ import annotations

import bisect
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

try:
    import hdf5plugin  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "hdf5plugin is required to load voxel HDF5 datasets. "
        "Please install dependencies (e.g. `pip install -r requirements.txt`)."
    ) from exc

import h5py
import numpy as np
import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler


class ConcatIndices:
    """Map a global index to (dataset_idx, local_idx) for concatenated datasets."""

    def __init__(self, sizes: Sequence[int]):
        if len(sizes) == 0:
            raise ValueError("sizes must be non-empty")
        self.cumulative_sizes = np.cumsum(sizes)

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise ValueError(f"index out of bounds: idx={idx}, size={len(self)}")
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            return 0, idx
        return dataset_idx, idx - int(self.cumulative_sizes[dataset_idx - 1])


def _parse_manifest_paths(manifest_path: Path) -> list[Path]:
    """Accepts CSV/TXT manifests and returns the first column/token as a path."""
    rows: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if len(raw) == 0:
                continue
            first = raw[0].strip()
            if not first:
                continue
            if first.startswith("#"):
                continue
            # Handle plain whitespace-delimited manifests too.
            token = first.split()[0]
            p = Path(token)
            if not p.is_absolute():
                p = (manifest_path.parent / p).resolve()
            rows.append(p)
    return rows


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            if "voxels" not in h5f:
                return False
            ds = h5f["voxels"]
            return ds.ndim == 4 and ds.shape[0] > 0
    except Exception:
        return False


def _discover_h5_files(
    data_path: Path,
    *,
    file_pattern: str,
    recursive: bool,
    require_voxels_key: bool,
) -> list[Path]:
    suffix = data_path.suffix.lower()
    files: list[Path] = []

    if suffix in {".csv", ".txt"}:
        files = _parse_manifest_paths(data_path)
    elif suffix in {".h5", ".hdf5"} and data_path.is_file():
        files = [data_path.resolve()]
    elif data_path.is_dir():
        iterator = data_path.rglob(file_pattern) if recursive else data_path.glob(file_pattern)
        files = sorted(p.resolve() for p in iterator if p.is_file())
    else:
        raise FileNotFoundError(f"Unsupported data path: {data_path}")

    # If file_pattern is broad (*.h5), keep only preprocessed voxel H5.
    if require_voxels_key:
        files = [p for p in files if _is_voxel_h5(p)]
    return files


def _to_cthw_tensor(buffer: np.ndarray) -> torch.Tensor:
    """
    Convert [T, H, W, C] -> [C, T, H, W].
    """
    if buffer.ndim != 4:
        raise ValueError(f"buffer must be 4D [T,H,W,C], got shape={buffer.shape}")
    tensor = torch.from_numpy(buffer).to(torch.float32)
    return tensor.permute(3, 0, 1, 2).contiguous()


def _is_sequence_but_not_str(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Path))


def _coerce_optional_float(value, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        value = text
    return float(value)


def _coerce_optional_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        value = text
    coerced = int(value)
    if coerced <= 0:
        raise ValueError(f"{field_name} values must be > 0 when provided")
    return coerced


def _expand_optional_per_dataset(value, num_datasets: int, field_name: str) -> list[float | None]:
    if num_datasets <= 0:
        raise ValueError("num_datasets must be > 0")
    if value is None:
        return [None] * num_datasets
    if _is_sequence_but_not_str(value):
        values = list(value)
        if len(values) == 0:
            return [None] * num_datasets
        if len(values) == 1 and num_datasets > 1:
            coerced = _coerce_optional_float(values[0], field_name)
            return [coerced] * num_datasets
        if len(values) != num_datasets:
            raise ValueError(f"{field_name} length must match number of datasets ({num_datasets})")
        return [_coerce_optional_float(v, field_name) for v in values]
    coerced = _coerce_optional_float(value, field_name)
    return [coerced] * num_datasets


def _expand_optional_int_per_dataset(value, num_datasets: int, field_name: str) -> list[int | None]:
    if num_datasets <= 0:
        raise ValueError("num_datasets must be > 0")
    if value is None:
        return [None] * num_datasets
    if _is_sequence_but_not_str(value):
        values = list(value)
        if len(values) == 0:
            return [None] * num_datasets
        if len(values) == 1 and num_datasets > 1:
            coerced = _coerce_optional_int(values[0], field_name)
            return [coerced] * num_datasets
        if len(values) != num_datasets:
            raise ValueError(f"{field_name} length must match number of datasets ({num_datasets})")
        return [_coerce_optional_int(v, field_name) for v in values]
    coerced = _coerce_optional_int(value, field_name)
    return [coerced] * num_datasets


def _normalize_voxel_time_mode(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "channel": "channels",
        "channels": "channels",
        "bins_as_channels": "channels",
        "t_bins_as_channels": "channels",
        "temporal": "temporal_bins",
        "temporal_bins": "temporal_bins",
        "bins_as_frames": "temporal_bins",
        "t_bins_as_frames": "temporal_bins",
        "tbin_as_time": "temporal_bins",
        "tbins_as_time": "temporal_bins",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported voxel_time_mode={value!r}. "
            "Use 'channels' or 'temporal_bins'."
        )
    return aliases[key]


def _h5_attr_int(h5f: h5py.File, key: str) -> int | None:
    if key not in h5f.attrs:
        return None
    value = h5f.attrs.get(key)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    try:
        return int(value)
    except Exception:
        return None


class EventVideoDataset(torch.utils.data.Dataset):
    """
    Stage-1 event representation dataset for JEPA-style pretraining.

    Output format is intentionally aligned with vjepa2 VideoDataset:
      (buffer, label, clip_indices)
    where `buffer` is a list of clips, each transformed to [C, T, H, W].
    """

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        *,
        datasets_weights: Sequence[float] | None = None,
        frames_per_clip: int = 8,
        dataset_fpcs: Sequence[int] | None = None,
        source_window_fpcs: Sequence[int] | None = None,
        voxel_time_mode: str = "channels",
        voxel_temporal_bins=None,
        frame_step: int = 1,
        fps=None,
        num_clips: int = 1,
        transform: Callable | None = None,
        shared_transform: Callable | None = None,
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        file_pattern: str = "*.h5",
        recursive: bool = True,
        require_voxels_key: bool = True,
        max_open_h5_files: int = 32,
        activity_filter_enabled: bool = False,
        activity_filter_min_clip_mean_active_pixel_ratio=None,
        activity_filter_min_clip_mean_activity_score=None,
        activity_filter_min_clip_active_window_ratio=None,
        activity_filter_active_window_threshold=None,
    ):
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        if len(data_paths) == 0:
            raise ValueError("data_paths must be non-empty")
        if frame_step <= 0:
            raise ValueError("frame_step must be > 0")
        if num_clips <= 0:
            raise ValueError("num_clips must be > 0")
        if frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be > 0")

        self.data_paths = [Path(p).expanduser() for p in data_paths]
        self.datasets_weights = datasets_weights
        self.frame_step = int(frame_step)
        self.num_clips = int(num_clips)
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = bool(random_clip_sampling)
        self.allow_clip_overlap = bool(allow_clip_overlap)
        self.max_open_h5_files = max(1, int(max_open_h5_files))
        self.activity_filter_enabled = bool(activity_filter_enabled)

        if dataset_fpcs is None:
            self.dataset_fpcs = [int(frames_per_clip) for _ in self.data_paths]
        else:
            if len(dataset_fpcs) != len(self.data_paths):
                raise ValueError("dataset_fpcs length must match data_paths length")
            self.dataset_fpcs = [int(v) for v in dataset_fpcs]
        if any(fpc <= 0 for fpc in self.dataset_fpcs):
            raise ValueError("All dataset_fpcs must be > 0")

        self.voxel_time_mode = _normalize_voxel_time_mode(voxel_time_mode)
        self.expected_voxel_temporal_bins = _expand_optional_int_per_dataset(
            voxel_temporal_bins,
            num_datasets=len(self.data_paths),
            field_name="voxel_temporal_bins",
        )
        if source_window_fpcs is None:
            if self.voxel_time_mode == "temporal_bins":
                if any(v is None for v in self.expected_voxel_temporal_bins):
                    raise ValueError(
                        "source_window_fpcs must be set when voxel_time_mode='temporal_bins' "
                        "unless voxel_temporal_bins is provided for every dataset."
                    )
                self.source_window_fpcs = []
                for model_fpc, bins in zip(self.dataset_fpcs, self.expected_voxel_temporal_bins):
                    assert bins is not None
                    if model_fpc % bins != 0:
                        raise ValueError(
                            "dataset_fpcs must be divisible by voxel_temporal_bins when "
                            "source_window_fpcs is omitted. "
                            f"Got dataset_fpc={model_fpc}, voxel_temporal_bins={bins}."
                        )
                    self.source_window_fpcs.append(model_fpc // bins)
            else:
                self.source_window_fpcs = list(self.dataset_fpcs)
        else:
            if len(source_window_fpcs) != len(self.data_paths):
                raise ValueError("source_window_fpcs length must match data_paths length")
            self.source_window_fpcs = [int(v) for v in source_window_fpcs]
        if any(fpc <= 0 for fpc in self.source_window_fpcs):
            raise ValueError("All source_window_fpcs must be > 0")

        self.target_fps_per_dataset = _expand_optional_per_dataset(
            fps,
            num_datasets=len(self.data_paths),
            field_name="fps",
        )
        for resolved_fps in self.target_fps_per_dataset:
            if resolved_fps is not None and float(resolved_fps) <= 0.0:
                raise ValueError("fps values must be > 0 when provided")

        self.activity_filter_min_clip_mean_active_pixel_ratio = _expand_optional_per_dataset(
            activity_filter_min_clip_mean_active_pixel_ratio,
            num_datasets=len(self.data_paths),
            field_name="activity_filter_min_clip_mean_active_pixel_ratio",
        )
        self.activity_filter_min_clip_mean_activity_score = _expand_optional_per_dataset(
            activity_filter_min_clip_mean_activity_score,
            num_datasets=len(self.data_paths),
            field_name="activity_filter_min_clip_mean_activity_score",
        )
        self.activity_filter_min_clip_active_window_ratio = _expand_optional_per_dataset(
            activity_filter_min_clip_active_window_ratio,
            num_datasets=len(self.data_paths),
            field_name="activity_filter_min_clip_active_window_ratio",
        )
        self.activity_filter_active_window_threshold = _expand_optional_per_dataset(
            activity_filter_active_window_threshold,
            num_datasets=len(self.data_paths),
            field_name="activity_filter_active_window_threshold",
        )
        for idx in range(len(self.data_paths)):
            if self.activity_filter_min_clip_active_window_ratio[idx] is not None:
                threshold = self.activity_filter_active_window_threshold[idx]
                if threshold is None:
                    self.activity_filter_active_window_threshold[idx] = 1e-6

        samples: list[str] = []
        labels: list[int] = []
        self.num_samples_per_dataset: list[int] = []
        for path in self.data_paths:
            dataset_files = _discover_h5_files(
                path,
                file_pattern=file_pattern,
                recursive=recursive,
                require_voxels_key=require_voxels_key,
            )
            if len(dataset_files) == 0:
                raise FileNotFoundError(f"No voxel H5 files found in: {path}")
            samples += [str(p) for p in dataset_files]
            labels += [0] * len(dataset_files)
            self.num_samples_per_dataset.append(len(dataset_files))

        self.samples = samples
        self.labels = labels
        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.sample_weights: list[float] | None = None
        if self.datasets_weights is not None:
            if len(self.datasets_weights) != len(self.num_samples_per_dataset):
                raise ValueError("datasets_weights length must match number of datasets")
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [float(dw) / float(ns)] * int(ns)

        # Worker-local lazy cache. We intentionally avoid sharing handles.
        self._h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._activity_cache: OrderedDict[str, tuple[np.ndarray | None, np.ndarray | None]] = OrderedDict()
        self._timestamp_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._valid_clip_start_cache: dict[tuple[str, int, int, int, int, int], np.ndarray] = {}
        self._sampling_step_cache: dict[tuple[str, int, int], int] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = OrderedDict()
        state["_timestamp_cache"] = OrderedDict()
        state["_sampling_step_cache"] = {}
        return state

    def __del__(self):
        cache = getattr(self, "_h5_cache", None)
        if isinstance(cache, dict):
            for h5f in cache.values():
                try:
                    h5f.close()
                except Exception:
                    pass
            cache.clear()
        activity_cache = getattr(self, "_activity_cache", None)
        if isinstance(activity_cache, dict):
            activity_cache.clear()
        timestamp_cache = getattr(self, "_timestamp_cache", None)
        if isinstance(timestamp_cache, dict):
            timestamp_cache.clear()
        valid_cache = getattr(self, "_valid_clip_start_cache", None)
        if isinstance(valid_cache, dict):
            valid_cache.clear()
        sampling_step_cache = getattr(self, "_sampling_step_cache", None)
        if isinstance(sampling_step_cache, dict):
            sampling_step_cache.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def _get_h5(self, sample_path: str) -> h5py.File:
        h5f = self._h5_cache.get(sample_path)
        if h5f is not None:
            self._h5_cache.move_to_end(sample_path, last=True)
            return h5f

        h5f = h5py.File(sample_path, "r")
        self._h5_cache[sample_path] = h5f
        self._h5_cache.move_to_end(sample_path, last=True)

        while len(self._h5_cache) > self.max_open_h5_files:
            old_key, old_h5 = self._h5_cache.popitem(last=False)
            try:
                old_h5.close()
            except Exception:
                pass
            self._activity_cache.pop(old_key, None)
        return h5f

    def _dataset_activity_filter_enabled(self, dataset_idx: int) -> bool:
        if not self.activity_filter_enabled:
            return False
        return any(
            (
                self.activity_filter_min_clip_mean_active_pixel_ratio[dataset_idx] is not None,
                self.activity_filter_min_clip_mean_activity_score[dataset_idx] is not None,
                self.activity_filter_min_clip_active_window_ratio[dataset_idx] is not None,
            )
        )

    def _get_anchor_timestamps_us(
        self,
        sample_path: str,
        total_windows: int,
    ) -> np.ndarray | None:
        if sample_path in self._timestamp_cache:
            cached = self._timestamp_cache[sample_path]
            self._timestamp_cache.move_to_end(sample_path, last=True)
            return cached

        h5f = self._get_h5(sample_path)
        timestamps = None
        for key in ("anchor_timestamp_us", "anchor_rel_timestamp_us"):
            if key in h5f:
                arr = np.asarray(h5f[key], dtype=np.int64).reshape(-1)
                if arr.size > 0:
                    timestamps = arr
                    break

        if timestamps is None:
            stride_us = int(h5f.attrs.get("stride_time_us", 0) or 0)
            if stride_us <= 0:
                stride_us = int(h5f.attrs.get("accum_time_us", 0) or 0)
            if stride_us > 0 and total_windows > 0:
                timestamps = np.arange(total_windows, dtype=np.int64) * np.int64(stride_us)

        if timestamps is not None:
            if timestamps.shape[0] < total_windows:
                timestamps = None
            else:
                timestamps = timestamps[:total_windows].astype(np.int64, copy=False)
                if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
                    timestamps = None

        self._timestamp_cache[sample_path] = timestamps
        self._timestamp_cache.move_to_end(sample_path, last=True)
        while len(self._timestamp_cache) > self.max_open_h5_files:
            self._timestamp_cache.popitem(last=False)
        return timestamps

    def _resolve_sampling_step(
        self,
        *,
        sample_path: str,
        dataset_idx: int,
        total_windows: int,
    ) -> int:
        target_fps = self.target_fps_per_dataset[dataset_idx]
        if target_fps is None:
            return self.frame_step

        cache_key = (sample_path, int(dataset_idx), int(total_windows))
        cached = self._sampling_step_cache.get(cache_key)
        if cached is not None:
            return int(cached)

        median_delta_us = None
        timestamps = self._get_anchor_timestamps_us(sample_path, total_windows)
        if timestamps is not None and timestamps.size >= 2:
            deltas_us = np.diff(timestamps)
            deltas_us = deltas_us[deltas_us > 0]
            if deltas_us.size > 0:
                median_delta_us = float(np.median(deltas_us))

        if median_delta_us is None:
            h5f = self._get_h5(sample_path)
            stride_us = int(h5f.attrs.get("stride_time_us", 0) or 0)
            if stride_us <= 0:
                stride_us = int(h5f.attrs.get("accum_time_us", 0) or 0)
            if stride_us > 0:
                median_delta_us = float(stride_us)

        if median_delta_us is None or median_delta_us <= 0.0:
            sampling_step = int(self.frame_step)
        else:
            source_fps = int(np.ceil(1_000_000.0 / median_delta_us))
            sampling_step = max(int(source_fps // float(target_fps)), 1)

        self._sampling_step_cache[cache_key] = int(sampling_step)
        return int(sampling_step)

    def _get_activity_arrays(self, sample_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        cached = self._activity_cache.get(sample_path)
        if cached is not None:
            self._activity_cache.move_to_end(sample_path, last=True)
            return cached

        h5f = self._get_h5(sample_path)
        active_ratio = None
        activity_score = None
        if "window_active_pixel_ratio" in h5f:
            active_ratio = np.asarray(h5f["window_active_pixel_ratio"], dtype=np.float32).reshape(-1)
        if "window_activity_score" in h5f:
            activity_score = np.asarray(h5f["window_activity_score"], dtype=np.float32).reshape(-1)

        cached = (active_ratio, activity_score)
        self._activity_cache[sample_path] = cached
        self._activity_cache.move_to_end(sample_path, last=True)
        while len(self._activity_cache) > self.max_open_h5_files:
            self._activity_cache.popitem(last=False)
        return cached

    @staticmethod
    def _fit_indices_length(indices: np.ndarray, target_len: int, last_valid: int) -> np.ndarray:
        if indices.size >= target_len:
            pos = np.linspace(0, indices.size - 1, num=target_len, dtype=np.float64)
            pos = np.round(pos).astype(np.int64)
            return indices[pos]
        pad = np.full((target_len - indices.size,), int(last_valid), dtype=np.int64)
        return np.concatenate([indices, pad], axis=0)

    def _sample_clip_in_segment(
        self,
        *,
        sample_path: str,
        dataset_idx: int,
        segment_start: int,
        segment_length: int,
        total_windows: int,
        fpc: int,
    ) -> np.ndarray | None:
        if total_windows <= 0:
            return np.zeros((fpc,), dtype=np.int64)

        if segment_length <= 0:
            anchor = min(max(segment_start, 0), total_windows - 1)
            return np.full((fpc,), anchor, dtype=np.int64)

        sampling_step = self._resolve_sampling_step(
            sample_path=sample_path,
            dataset_idx=dataset_idx,
            total_windows=total_windows,
        )

        def _candidate_local_starts() -> np.ndarray:
            clip_span = max(1, fpc * sampling_step)
            if segment_length > clip_span:
                max_local_start = segment_length - clip_span
                return np.arange(0, max_local_start + 1, dtype=np.int64)
            return np.array([0], dtype=np.int64)

        def _indices_from_local_start(local_start: int) -> np.ndarray:
            clip_span = max(1, fpc * sampling_step)
            if segment_length > clip_span:
                local_indices = np.arange(
                    int(local_start),
                    int(local_start) + clip_span,
                    sampling_step,
                    dtype=np.int64,
                )
            else:
                local_indices = np.arange(0, segment_length, sampling_step, dtype=np.int64)
            if local_indices.size == 0:
                local_indices = np.array([0], dtype=np.int64)
            local_indices = self._fit_indices_length(
                local_indices,
                target_len=fpc,
                last_valid=max(0, segment_length - 1),
            )
            global_indices = segment_start + local_indices
            np.clip(global_indices, 0, total_windows - 1, out=global_indices)
            return global_indices.astype(np.int64, copy=False)

        def _passes_activity_filter(indices: np.ndarray) -> bool:
            if not self._dataset_activity_filter_enabled(dataset_idx):
                return True

            active_ratio, activity_score = self._get_activity_arrays(sample_path)
            min_mean_active = self.activity_filter_min_clip_mean_active_pixel_ratio[dataset_idx]
            min_mean_score = self.activity_filter_min_clip_mean_activity_score[dataset_idx]
            min_active_window_ratio = self.activity_filter_min_clip_active_window_ratio[dataset_idx]
            active_window_threshold = self.activity_filter_active_window_threshold[dataset_idx]

            if min_mean_active is not None:
                if active_ratio is None or active_ratio.shape[0] < total_windows:
                    return False
                if float(np.mean(active_ratio[indices])) < float(min_mean_active):
                    return False

            if min_mean_score is not None:
                if activity_score is None or activity_score.shape[0] < total_windows:
                    return False
                if float(np.mean(activity_score[indices])) < float(min_mean_score):
                    return False

            if min_active_window_ratio is not None:
                if active_ratio is None or active_ratio.shape[0] < total_windows:
                    return False
                threshold = 0.0 if active_window_threshold is None else float(active_window_threshold)
                active_mask = active_ratio[indices] > threshold
                if float(np.mean(active_mask.astype(np.float32))) < float(min_active_window_ratio):
                    return False

            return True

        candidate_local_starts = _candidate_local_starts()
        if self._dataset_activity_filter_enabled(dataset_idx):
            cache_key = (
                sample_path,
                int(dataset_idx),
                int(segment_start),
                int(segment_length),
                int(fpc),
                int(sampling_step),
            )
            valid_local_starts = self._valid_clip_start_cache.get(cache_key)
            if valid_local_starts is None:
                valid = []
                for local_start in candidate_local_starts.tolist():
                    candidate_indices = _indices_from_local_start(int(local_start))
                    if _passes_activity_filter(candidate_indices):
                        valid.append(int(local_start))
                valid_local_starts = np.asarray(valid, dtype=np.int64)
                self._valid_clip_start_cache[cache_key] = valid_local_starts
            if valid_local_starts.size == 0:
                return None
            if self.random_clip_sampling:
                chosen_local_start = int(valid_local_starts[np.random.randint(0, valid_local_starts.size)])
            else:
                chosen_local_start = int(valid_local_starts[valid_local_starts.size // 2])
            return _indices_from_local_start(chosen_local_start)

        clip_span = max(1, fpc * sampling_step)
        if segment_length > clip_span:
            max_local_start = segment_length - clip_span
            if self.random_clip_sampling:
                local_start = int(np.random.randint(0, max_local_start + 1))
            else:
                local_start = max_local_start // 2
            local_indices = np.arange(
                local_start,
                local_start + clip_span,
                sampling_step,
                dtype=np.int64,
            )
        else:
            local_indices = np.arange(0, segment_length, sampling_step, dtype=np.int64)

        if local_indices.size == 0:
            local_indices = np.array([0], dtype=np.int64)
        local_indices = self._fit_indices_length(
            local_indices,
            target_len=fpc,
            last_valid=max(0, segment_length - 1),
        )

        global_indices = segment_start + local_indices
        np.clip(global_indices, 0, total_windows - 1, out=global_indices)
        return global_indices.astype(np.int64, copy=False)

    def _sample_clip_indices(self, *, sample_path: str, dataset_idx: int, total_windows: int, fpc: int) -> list[np.ndarray] | None:
        if self.num_clips == 1:
            indices = self._sample_clip_in_segment(
                sample_path=sample_path,
                dataset_idx=dataset_idx,
                segment_start=0,
                segment_length=total_windows,
                total_windows=total_windows,
                fpc=fpc,
            )
            if indices is None:
                return None
            return [indices]

        all_clip_indices: list[np.ndarray] = []
        if not self.allow_clip_overlap:
            partition_len = max(1, total_windows // self.num_clips)
            for clip_id in range(self.num_clips):
                start = clip_id * partition_len
                end = total_windows if clip_id == self.num_clips - 1 else min(total_windows, (clip_id + 1) * partition_len)
                length = max(1, end - start)
                indices = self._sample_clip_in_segment(
                    sample_path=sample_path,
                    dataset_idx=dataset_idx,
                    segment_start=start,
                    segment_length=length,
                    total_windows=total_windows,
                    fpc=fpc,
                )
                if indices is None:
                    return None
                all_clip_indices.append(indices)
        else:
            clip_span = max(1, fpc * self.frame_step)
            if self.num_clips == 1:
                starts = np.array([0], dtype=np.int64)
            elif total_windows <= clip_span:
                starts = np.linspace(0, max(0, total_windows - 1), num=self.num_clips, dtype=np.float64)
                starts = np.round(starts).astype(np.int64)
            else:
                starts = np.linspace(0, total_windows - clip_span, num=self.num_clips, dtype=np.float64)
                starts = np.round(starts).astype(np.int64)
            for start in starts.tolist():
                seg_len = min(total_windows - start, clip_span)
                indices = self._sample_clip_in_segment(
                    sample_path=sample_path,
                    dataset_idx=dataset_idx,
                    segment_start=int(start),
                    segment_length=int(seg_len),
                    total_windows=total_windows,
                    fpc=fpc,
                )
                if indices is None:
                    return None
                all_clip_indices.append(indices)
        return all_clip_indices

    @staticmethod
    def _read_windows(voxels_ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            raise ValueError("indices cannot be empty")

        if indices.size > 1 and np.all(indices[1:] - indices[:-1] == 1):
            start = int(indices[0])
            end = int(indices[-1]) + 1
            arr = np.asarray(voxels_ds[start:end])
        else:
            arr = np.stack([np.asarray(voxels_ds[int(i)]) for i in indices], axis=0)

        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]
        if arr.ndim != 4:
            raise ValueError(f"Unexpected voxel window shape: {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def _infer_temporal_layout(
        self,
        h5f: h5py.File,
        *,
        channels: int,
        dataset_idx: int,
    ) -> tuple[int, int]:
        expected_bins = self.expected_voxel_temporal_bins[dataset_idx]
        t_bins = _h5_attr_int(h5f, "t_bins")
        if t_bins is None or t_bins <= 0:
            t_bins = expected_bins
        if t_bins is None or t_bins <= 0:
            raise ValueError(
                "Could not infer voxel t_bins from H5 attrs. "
                "Set data.voxel_temporal_bins for temporal-bin training."
            )
        if expected_bins is not None and int(t_bins) != int(expected_bins):
            raise ValueError(
                f"H5 t_bins={t_bins} does not match configured voxel_temporal_bins={expected_bins}."
            )

        split_polarity = _h5_attr_int(h5f, "split_polarity")
        if int(t_bins) == 1:
            return 1, int(channels)

        if split_polarity is not None and bool(split_polarity):
            if channels % int(t_bins) != 0:
                raise ValueError(f"channels={channels} is not divisible by t_bins={t_bins}")
            polarity_channels = channels // int(t_bins)
        elif channels == int(t_bins):
            polarity_channels = 1
        elif channels % int(t_bins) == 0:
            polarity_channels = channels // int(t_bins)
        else:
            raise ValueError(
                f"Cannot reshape voxel channels={channels} into temporal bins={t_bins}. "
                "Check preprocessing t_bins/split_polarity and model.in_chans."
            )
        return int(t_bins), int(polarity_channels)

    def _windows_to_model_frames(
        self,
        h5f: h5py.File,
        windows_tchw: np.ndarray,
        *,
        dataset_idx: int,
    ) -> tuple[np.ndarray, int]:
        if self.voxel_time_mode == "channels":
            return windows_tchw, 1

        if windows_tchw.ndim != 4:
            raise ValueError(f"Expected [T,C,H,W] windows, got shape={windows_tchw.shape}")
        source_t, channels, height, width = windows_tchw.shape
        t_bins, polarity_channels = self._infer_temporal_layout(
            h5f,
            channels=int(channels),
            dataset_idx=dataset_idx,
        )
        if t_bins == 1:
            return windows_tchw, 1

        expected_channels = t_bins * polarity_channels
        if channels != expected_channels:
            raise ValueError(
                f"Expected channels=t_bins*polarity_channels={expected_channels}, got {channels}."
            )

        groups = [
            windows_tchw[:, i * t_bins : (i + 1) * t_bins]
            for i in range(polarity_channels)
        ]
        # [source_T, t_bins, polarity_channels, H, W]
        expanded = np.stack(groups, axis=2)
        expanded = expanded.reshape(source_t * t_bins, polarity_channels, height, width)
        return expanded.astype(np.float32, copy=False), int(t_bins)

    def _expand_clip_indices(
        self,
        clip_indices: list[np.ndarray],
        *,
        bins_per_window: int,
    ) -> list[np.ndarray]:
        if self.voxel_time_mode == "channels" or bins_per_window <= 1:
            return [ci.astype(np.int32, copy=False) for ci in clip_indices]

        expanded = []
        for indices in clip_indices:
            frame_indices = []
            for window_idx in indices.tolist():
                base = int(window_idx) * int(bins_per_window)
                frame_indices.extend(base + bin_idx for bin_idx in range(int(bins_per_window)))
            expanded.append(np.asarray(frame_indices, dtype=np.int32))
        return expanded

    def get_item_event(self, index: int):
        sample_path = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        source_fpc = self.source_window_fpcs[dataset_idx]
        model_fpc = self.dataset_fpcs[dataset_idx]

        h5f = self._get_h5(sample_path)
        if "voxels" not in h5f:
            return None
        voxels_ds = h5f["voxels"]
        if voxels_ds.ndim != 4 or voxels_ds.shape[0] <= 0:
            return None

        total_windows = int(voxels_ds.shape[0])
        clip_indices = self._sample_clip_indices(
            sample_path=sample_path,
            dataset_idx=dataset_idx,
            total_windows=total_windows,
            fpc=source_fpc,
        )
        if clip_indices is None:
            return None
        all_indices = np.concatenate(clip_indices, axis=0).astype(np.int64, copy=False)
        windows = self._read_windows(voxels_ds=voxels_ds, indices=all_indices)  # [T,C,H,W]
        windows, bins_per_window = self._windows_to_model_frames(
            h5f,
            windows,
            dataset_idx=dataset_idx,
        )
        expected_total_frames = int(model_fpc) * int(self.num_clips)
        if int(windows.shape[0]) != expected_total_frames:
            raise ValueError(
                "Model temporal length mismatch after voxel time conversion: "
                f"got {windows.shape[0]} frames for {self.num_clips} clips, "
                f"expected {expected_total_frames}. "
                f"source_window_fpcs={source_fpc}, dataset_fpcs={model_fpc}, "
                f"bins_per_window={bins_per_window}, voxel_time_mode={self.voxel_time_mode}."
            )
        windows = np.transpose(windows, (0, 2, 3, 1))  # [T,H,W,C]

        if self.shared_transform is not None:
            windows = self.shared_transform(windows)

        split_clips = [windows[i * model_fpc : (i + 1) * model_fpc] for i in range(self.num_clips)]
        if self.transform is not None:
            split_clips = [self.transform(clip) for clip in split_clips]
        else:
            split_clips = [_to_cthw_tensor(clip) for clip in split_clips]

        label = self.labels[index]
        clip_indices = self._expand_clip_indices(clip_indices, bins_per_window=bins_per_window)
        return split_clips, label, clip_indices

    def __getitem__(self, index: int):
        num_trials = 8
        for _ in range(num_trials):
            loaded = self.get_item_event(index)
            if loaded is not None:
                return loaded
            index = int(np.random.randint(0, len(self.samples)))
        raise RuntimeError(f"Failed to load event sample after {num_trials} retries.")


def make_eventdataset(
    data_paths: str | Path | Sequence[str | Path],
    batch_size: int,
    *,
    frames_per_clip: int = 8,
    dataset_fpcs: Sequence[int] | None = None,
    source_window_fpcs: Sequence[int] | None = None,
    voxel_time_mode: str = "channels",
    voxel_temporal_bins=None,
    frame_step: int = 1,
    fps=None,
    num_clips: int = 1,
    random_clip_sampling: bool = True,
    allow_clip_overlap: bool = False,
    transform: Callable | None = None,
    shared_transform: Callable | None = None,
    rank: int = 0,
    world_size: int = 1,
    datasets_weights: Sequence[float] | None = None,
    collator: Callable | None = None,
    drop_last: bool = True,
    num_workers: int = 10,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = None,
    max_open_h5_files: int = 32,
    file_pattern: str = "*.h5",
    recursive: bool = True,
    require_voxels_key: bool = True,
    activity_filter_enabled: bool = False,
    activity_filter_min_clip_mean_active_pixel_ratio=None,
    activity_filter_min_clip_mean_activity_score=None,
    activity_filter_min_clip_active_window_ratio=None,
    activity_filter_active_window_threshold=None,
):
    dataset = EventVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        source_window_fpcs=source_window_fpcs,
        voxel_time_mode=voxel_time_mode,
        voxel_temporal_bins=voxel_temporal_bins,
        frame_step=frame_step,
        fps=fps,
        num_clips=num_clips,
        transform=transform,
        shared_transform=shared_transform,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        file_pattern=file_pattern,
        recursive=recursive,
        require_voxels_key=require_voxels_key,
        max_open_h5_files=max_open_h5_files,
        activity_filter_enabled=activity_filter_enabled,
        activity_filter_min_clip_mean_active_pixel_ratio=activity_filter_min_clip_mean_active_pixel_ratio,
        activity_filter_min_clip_mean_activity_score=activity_filter_min_clip_mean_activity_score,
        activity_filter_min_clip_active_window_ratio=activity_filter_min_clip_active_window_ratio,
        activity_filter_active_window_threshold=activity_filter_active_window_threshold,
    )

    if datasets_weights is not None:
        sampler = DistributedWeightedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    dataloader_kwargs = dict(
        dataset=dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    data_loader = torch.utils.data.DataLoader(**dataloader_kwargs)
    return dataset, data_loader, sampler
