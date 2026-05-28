from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _required_path(value: object | None, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"{field_name} is required")
    text = str(value).strip()
    if len(text) == 0:
        raise ValueError(f"{field_name} is required")
    return Path(text)


def _optional_path(value: object | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) == 0 or text.lower() == "null":
        return None
    return Path(text)


def _list_of_str(value: object, field_name: str) -> list[str]:
    if isinstance(value, (list, tuple, ListConfig)):
        out = [str(v) for v in value]
        if len(out) == 0:
            raise ValueError(f"{field_name} must not be empty")
        return out
    if value is None:
        raise ValueError(f"{field_name} must not be empty")
    text = str(value).strip()
    if len(text) == 0:
        raise ValueError(f"{field_name} must not be empty")
    return [text]


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _cfg_get(cfg: DictConfig, key: str, default):
    return cfg.get(key, default)


def _run_dsec(cfg: DictConfig) -> None:
    from scripts.preprocess.preprocess_dsec import process_dataset_root

    stride_time = int(cfg.accum_time) if cfg.stride_time is None else int(cfg.stride_time)
    process_dataset_root(
        dataset_root=_required_path(cfg.dataset_root, "dataset.dataset_root"),
        splits=_list_of_str(cfg.splits, "dataset.splits"),
        output_name=None if cfg.output_name is None else str(cfg.output_name),
        output_suffix=None if cfg.output_suffix is None else str(cfg.output_suffix),
        output_subdir=None if cfg.output_subdir is None else str(cfg.output_subdir),
        overwrite=bool(cfg.overwrite),
        output_root=_optional_path(cfg.output_root),
        height=int(cfg.input_height),
        width=int(cfg.input_width),
        downsample_factor=int(cfg.downsample_factor),
        t_bins=int(cfg.t_bins),
        split_polarity=bool(cfg.split_polarity),
        accum_time=int(cfg.accum_time),
        stride_time=stride_time,
        start_time_us=_optional_int(cfg.start_time_us),
        window_mode=str(cfg.window_mode),
        image_root=_optional_path(cfg.image_root),
        normalize=bool(cfg.normalize),
        output_dtype=str(cfg.output_dtype),
        use_trilinear=bool(cfg.use_trilinear),
        representation=str(_cfg_get(cfg, "representation", "voxel_grid")),
        event_image_percentile=float(_cfg_get(cfg, "event_image_percentile", 99.0)),
        save_mp4=bool(_cfg_get(cfg, "save_mp4", False)),
        mp4_fps=_optional_float(_cfg_get(cfg, "mp4_fps", None)),
        sync_segmentation=bool(cfg.sync_segmentation),
        segmentation_root=_optional_path(cfg.segmentation_root),
        segmentation_subdir=str(cfg.segmentation_subdir),
        segmentation_tolerance_us=int(cfg.segmentation_tolerance_us),
        activity_mode=str(cfg.activity_mode),
        activity_spatial_patch_size=int(cfg.activity_spatial_patch_size),
        activity_temporal_patch_size=int(cfg.activity_temporal_patch_size),
        tmp_suffix=str(cfg.tmp_suffix),
        num_processes=int(cfg.num_processes),
        split_chunk_duration_s=None if cfg.split_chunk_duration_s is None else float(cfg.split_chunk_duration_s),
        split_output_root=_optional_path(cfg.split_output_root),
        split_copy_batch_size=int(cfg.split_copy_batch_size),
        split_min_windows_per_chunk=int(cfg.split_min_windows_per_chunk),
        split_chunk_index_pad=int(cfg.split_chunk_index_pad),
        split_metadata_mode=str(cfg.split_metadata_mode),
        split_progress_interval_s=float(cfg.split_progress_interval_s),
        split_log_chunk_progress=bool(cfg.split_log_chunk_progress),
        split_log_dataset_progress=bool(cfg.split_log_dataset_progress),
        split_delete_source_after_success=bool(cfg.split_delete_source_after_success),
    )


def _run_1mpx(cfg: DictConfig) -> None:
    from scripts.preprocess.preprocess_1mpx import process_dataset_root

    stride_time = int(cfg.accum_time) if cfg.stride_time is None else int(cfg.stride_time)
    process_dataset_root(
        dataset_root=_required_path(cfg.dataset_root, "dataset.dataset_root"),
        splits=_list_of_str(cfg.splits, "dataset.splits"),
        output_suffix=str(cfg.output_suffix),
        output_subdir=None if cfg.output_subdir is None else str(cfg.output_subdir),
        overwrite=bool(cfg.overwrite),
        output_root=_optional_path(cfg.output_root),
        input_height=None if cfg.input_height is None else int(cfg.input_height),
        input_width=None if cfg.input_width is None else int(cfg.input_width),
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
        downsample_factor=int(cfg.downsample_factor),
        t_bins=int(cfg.t_bins),
        split_polarity=bool(cfg.split_polarity),
        accum_time=int(cfg.accum_time),
        stride_time=stride_time,
        start_time_us=_optional_int(cfg.start_time_us),
        normalize=bool(cfg.normalize),
        output_dtype=str(cfg.output_dtype),
        compression_level=int(cfg.compression_level),
        use_trilinear=bool(cfg.use_trilinear),
        representation=str(_cfg_get(cfg, "representation", "voxel_grid")),
        event_image_percentile=float(_cfg_get(cfg, "event_image_percentile", 99.0)),
        save_mp4=bool(_cfg_get(cfg, "save_mp4", False)),
        mp4_fps=_optional_float(_cfg_get(cfg, "mp4_fps", None)),
        activity_mode=str(cfg.activity_mode),
        activity_spatial_patch_size=int(cfg.activity_spatial_patch_size),
        activity_temporal_patch_size=int(cfg.activity_temporal_patch_size),
        writer_capacity_growth=str(cfg.writer_capacity_growth),
        rdcc_nbytes=int(cfg.rdcc_nbytes),
        rdcc_nslots=int(cfg.rdcc_nslots),
        rdcc_w0=float(cfg.rdcc_w0),
        recursive=bool(cfg.recursive),
        tmp_suffix=str(cfg.tmp_suffix),
        num_processes=int(cfg.num_processes),
        split_chunk_duration_s=None if cfg.split_chunk_duration_s is None else float(cfg.split_chunk_duration_s),
        split_output_root=_optional_path(cfg.split_output_root),
        split_copy_batch_size=int(cfg.split_copy_batch_size),
        split_min_windows_per_chunk=int(cfg.split_min_windows_per_chunk),
        split_chunk_index_pad=int(cfg.split_chunk_index_pad),
        split_metadata_mode=str(cfg.split_metadata_mode),
        split_progress_interval_s=float(cfg.split_progress_interval_s),
        split_log_chunk_progress=bool(cfg.split_log_chunk_progress),
        split_log_dataset_progress=bool(cfg.split_log_dataset_progress),
        split_delete_source_after_success=bool(cfg.split_delete_source_after_success),
    )


def _run_m3ed(cfg: DictConfig) -> None:
    from scripts.preprocess.preprocess_m3ed import process_dataset_root

    stride_time = int(cfg.accum_time) if cfg.stride_time is None else int(cfg.stride_time)
    process_dataset_root(
        dataset_root=_required_path(cfg.dataset_root, "dataset.dataset_root"),
        output_suffix=str(cfg.output_suffix),
        output_subdir=None if cfg.output_subdir is None else str(cfg.output_subdir),
        overwrite=bool(cfg.overwrite),
        output_root=_optional_path(cfg.output_root),
        input_height=None if cfg.input_height is None else int(cfg.input_height),
        input_width=None if cfg.input_width is None else int(cfg.input_width),
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
        downsample_factor=int(cfg.downsample_factor),
        t_bins=int(cfg.t_bins),
        split_polarity=bool(cfg.split_polarity),
        accum_time=int(cfg.accum_time),
        stride_time=stride_time,
        start_time_us=_optional_int(cfg.start_time_us),
        window_mode=str(cfg.window_mode),
        filter_known_semantic_sequences=bool(cfg.filter_known_semantic_sequences),
        filter_known_depth_sequences=bool(cfg.filter_known_depth_sequences),
        semantics_ts_source=str(cfg.semantics_ts_source),
        semantics_ts_divisor=int(cfg.semantics_ts_divisor),
        depth_ts_source=str(cfg.depth_ts_source),
        depth_ts_divisor=int(cfg.depth_ts_divisor),
        normalize=bool(cfg.normalize),
        output_dtype=str(cfg.output_dtype),
        use_trilinear=bool(cfg.use_trilinear),
        representation=str(_cfg_get(cfg, "representation", "voxel_grid")),
        event_image_percentile=float(_cfg_get(cfg, "event_image_percentile", 99.0)),
        save_mp4=bool(_cfg_get(cfg, "save_mp4", False)),
        mp4_fps=_optional_float(_cfg_get(cfg, "mp4_fps", None)),
        activity_mode=str(cfg.activity_mode),
        activity_spatial_patch_size=int(cfg.activity_spatial_patch_size),
        activity_temporal_patch_size=int(cfg.activity_temporal_patch_size),
        tmp_suffix=str(cfg.tmp_suffix),
        num_processes=int(cfg.num_processes),
        split_chunk_duration_s=None if cfg.split_chunk_duration_s is None else float(cfg.split_chunk_duration_s),
        split_output_root=_optional_path(cfg.split_output_root),
        split_copy_batch_size=int(cfg.split_copy_batch_size),
        split_min_windows_per_chunk=int(cfg.split_min_windows_per_chunk),
        split_chunk_index_pad=int(cfg.split_chunk_index_pad),
        split_metadata_mode=str(cfg.split_metadata_mode),
        split_progress_interval_s=float(cfg.split_progress_interval_s),
        split_log_chunk_progress=bool(cfg.split_log_chunk_progress),
        split_log_dataset_progress=bool(cfg.split_log_dataset_progress),
        split_delete_source_after_success=bool(cfg.split_delete_source_after_success),
    )


def _run_eventscape(cfg: DictConfig) -> None:
    from scripts.preprocess.preprocess_eventscape import process_dataset_root

    process_dataset_root(
        dataset_root=_required_path(cfg.dataset_root, "dataset.dataset_root"),
        splits=None if cfg.splits is None else _list_of_str(cfg.splits, "dataset.splits"),
        output_suffix=str(cfg.output_suffix),
        output_subdir=None if cfg.output_subdir is None else str(cfg.output_subdir),
        overwrite=bool(cfg.overwrite),
        output_root=_optional_path(cfg.output_root),
        input_height=int(cfg.input_height),
        input_width=int(cfg.input_width),
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
        downsample_factor=int(cfg.downsample_factor),
        t_bins=int(cfg.t_bins),
        split_polarity=bool(cfg.split_polarity),
        normalize=bool(cfg.normalize),
        output_dtype=str(cfg.output_dtype),
        use_trilinear=bool(cfg.use_trilinear),
        representation=str(_cfg_get(cfg, "representation", "voxel_grid")),
        event_image_percentile=float(_cfg_get(cfg, "event_image_percentile", 99.0)),
        save_mp4=bool(_cfg_get(cfg, "save_mp4", False)),
        mp4_fps=_optional_float(_cfg_get(cfg, "mp4_fps", None)),
        activity_mode=str(cfg.activity_mode),
        activity_spatial_patch_size=int(cfg.activity_spatial_patch_size),
        activity_temporal_patch_size=int(cfg.activity_temporal_patch_size),
        tmp_suffix=str(cfg.tmp_suffix),
        num_processes=int(cfg.num_processes),
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset
    target = str(dataset_cfg.name)

    if target == "dsec":
        _run_dsec(dataset_cfg)
    elif target == "1mpx":
        _run_1mpx(dataset_cfg)
    elif target == "m3ed":
        _run_m3ed(dataset_cfg)
    elif target == "eventscape":
        _run_eventscape(dataset_cfg)
    else:
        raise ValueError(f"unsupported dataset config: {target}")


if __name__ == "__main__":
    main()
