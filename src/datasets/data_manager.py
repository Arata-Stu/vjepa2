# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data="ImageNet",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    subset_file=None,
    clip_len=None,
    dataset_fpcs=None,
    source_window_fpcs=None,
    voxel_time_mode="channels",
    voxel_temporal_bins=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    datasets_weights=None,
    persistent_workers=False,
    prefetch_factor=None,
    max_open_h5_files=32,
    deterministic=True,
    log_dir=None,
    file_pattern="*.h5",
    recursive=True,
    require_voxels_key=True,
    activity_filter_enabled=False,
    activity_filter_min_clip_mean_active_pixel_ratio=None,
    activity_filter_min_clip_mean_activity_score=None,
    activity_filter_min_clip_active_window_ratio=None,
    activity_filter_active_window_threshold=None,
):
    if data.lower() == "imagenet":
        from src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "videodataset":
        from src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
        )

    elif data.lower() in {"eventdataset", "eventvoxel", "eventh5dataset"}:
        from src.datasets.event_dataset import make_eventdataset

        frames_per_clip = clip_len if clip_len is not None else 8
        frame_step = frame_sample_rate if frame_sample_rate is not None else 1
        dataset, data_loader, dist_sampler = make_eventdataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            dataset_fpcs=dataset_fpcs,
            source_window_fpcs=source_window_fpcs,
            voxel_time_mode=voxel_time_mode,
            voxel_temporal_bins=voxel_temporal_bins,
            frame_step=frame_step,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            transform=transform,
            shared_transform=shared_transform,
            rank=rank,
            world_size=world_size,
            datasets_weights=datasets_weights,
            collator=collator,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            max_open_h5_files=max_open_h5_files,
            file_pattern=file_pattern,
            recursive=recursive,
            require_voxels_key=require_voxels_key,
            activity_filter_enabled=activity_filter_enabled,
            activity_filter_min_clip_mean_active_pixel_ratio=activity_filter_min_clip_mean_active_pixel_ratio,
            activity_filter_min_clip_mean_activity_score=activity_filter_min_clip_mean_activity_score,
            activity_filter_min_clip_active_window_ratio=activity_filter_min_clip_active_window_ratio,
            activity_filter_active_window_threshold=activity_filter_active_window_threshold,
        )

    else:
        raise ValueError(
            f"Unsupported dataset type: {data}. "
            "Use one of: ImageNet, VideoDataset, EventDataset."
        )

    return (data_loader, dist_sampler)
