from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.event_dataset import EventVideoDataset
from src.datasets.event_transforms import make_event_transforms
from src.masks.multiseq_multiblock3d import MaskCollator

_RESAMPLING = getattr(Image, "Resampling", Image)


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_hw_tuple(value, field_name: str) -> tuple[int, int]:
    if isinstance(value, int):
        v = int(value)
        return (v, v)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{field_name} must be int or [H, W], got: {value}")


def _resolve_interpolation(mode: str):
    from torchvision.transforms import InterpolationMode

    lookup = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    key = str(mode).lower()
    if key not in lookup:
        raise ValueError(f"unsupported interpolation: {mode}")
    return lookup[key]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_to_uint8(arr: np.ndarray, *, percentile: float = 99.5) -> np.ndarray:
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


def _voxel_to_polarity_rgb(voxel: np.ndarray) -> np.ndarray:
    voxel = np.asarray(voxel, dtype=np.float32)
    if voxel.ndim != 3:
        raise ValueError(f"voxel must be [C,H,W], got shape={voxel.shape}")
    channels = int(voxel.shape[0])
    if channels >= 2 and channels % 2 == 0:
        half = channels // 2
        pos = np.abs(voxel[:half]).sum(axis=0)
        neg = np.abs(voxel[half:]).sum(axis=0)
    elif channels == 1:
        signed = voxel[0]
        pos = np.maximum(signed, 0.0)
        neg = np.maximum(-signed, 0.0)
    else:
        pos = np.abs(voxel[0::2]).sum(axis=0)
        neg = np.abs(voxel[1::2]).sum(axis=0)

    pos_u8 = _normalize_to_uint8(pos, percentile=99.5)
    neg_u8 = _normalize_to_uint8(neg, percentile=99.5)
    rgb = np.zeros((*pos_u8.shape, 3), dtype=np.uint8)
    rgb[..., 0] = pos_u8
    rgb[..., 2] = neg_u8
    rgb[..., 1] = np.minimum(pos_u8, neg_u8) // 2
    return rgb


def _indices_to_mask_volume(
    indices: torch.Tensor | np.ndarray,
    *,
    temporal_dim: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    total = int(temporal_dim) * int(grid_h) * int(grid_w)
    flat = np.zeros((total,), dtype=bool)
    index_arr = np.asarray(torch.as_tensor(indices).reshape(-1).cpu(), dtype=np.int64)
    if index_arr.size > 0:
        if np.any(index_arr < 0) or np.any(index_arr >= total):
            raise ValueError(
                f"mask indices out of bounds for total={total}: "
                f"min={int(index_arr.min())}, max={int(index_arr.max())}"
            )
        flat[index_arr] = True
    return flat.reshape(int(temporal_dim), int(grid_h), int(grid_w))


def _frame_to_temporal_index(frame_idx: int, *, num_frames: int, temporal_dim: int) -> int:
    if temporal_dim <= 1:
        return 0
    ratio = float(frame_idx) / float(max(1, num_frames))
    return min(int(temporal_dim - 1), int(math.floor(ratio * temporal_dim)))


def _patch_mask_to_pixel_mask(mask_2d: np.ndarray, *, height: int, width: int) -> np.ndarray:
    grid_h, grid_w = [int(v) for v in mask_2d.shape]
    repeat_h = max(1, int(math.ceil(float(height) / float(max(1, grid_h)))))
    repeat_w = max(1, int(math.ceil(float(width) / float(max(1, grid_w)))))
    up = np.repeat(np.repeat(mask_2d.astype(np.uint8), repeat_h, axis=0), repeat_w, axis=1)
    return up[:height, :width].astype(bool, copy=False)


def _overlay_patch_mask(
    base_rgb: np.ndarray,
    patch_mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    dim_alpha: float = 0.22,
    color_alpha: float = 0.78,
) -> np.ndarray:
    base = np.asarray(base_rgb, dtype=np.float32)
    out = base * float(max(0.0, min(1.0, dim_alpha)))
    pixel_mask = _patch_mask_to_pixel_mask(
        patch_mask,
        height=int(base_rgb.shape[0]),
        width=int(base_rgb.shape[1]),
    )
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    out[pixel_mask] = (
        base[pixel_mask] * (1.0 - float(color_alpha))
        + tint.reshape(3) * float(color_alpha)
    )
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _draw_patch_grid(
    image: Image.Image,
    *,
    grid_h: int,
    grid_w: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    for i in range(1, int(grid_h)):
        y = int(round(float(i) * float(height) / float(grid_h)))
        draw.line([(0, y), (width, y)], fill=color, width=1)
    for j in range(1, int(grid_w)):
        x = int(round(float(j) * float(width) / float(grid_w)))
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    return out


def _annotate_panel(image: Image.Image, text: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
    except Exception:
        text_w = max(1, len(text) * 6)
        text_h = 11
    pad = 3
    draw.rectangle(
        [(0, 0), (text_w + pad * 2, text_h + pad * 2)],
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def _make_strip(
    *,
    title: str,
    frames: list[Image.Image],
    frame_display_height: int,
) -> Image.Image:
    if len(frames) == 0:
        raise ValueError("frames cannot be empty")

    font = ImageFont.load_default()
    resized: list[Image.Image] = []
    for frame in frames:
        scale = max(1.0, float(frame_display_height) / float(max(1, frame.height)))
        new_size = (
            max(1, int(round(frame.width * scale))),
            max(1, int(round(frame.height * scale))),
        )
        resized.append(frame.resize(new_size, resample=_RESAMPLING.NEAREST))

    gap = 6
    pad = 8
    caption_h = 24
    strip_w = sum(img.width for img in resized) + gap * max(0, len(resized) - 1)
    strip_h = max(img.height for img in resized)
    canvas = Image.new(
        "RGB",
        (pad * 2 + strip_w, pad * 2 + caption_h + strip_h),
        color=(250, 250, 250),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(25, 30, 38), font=font)

    x = pad
    y = pad + caption_h
    for img in resized:
        canvas.paste(img, (x, y))
        x += img.width + gap
    return canvas


def _stack_images(images: list[Image.Image], *, bg_color: tuple[int, int, int] = (245, 245, 245)) -> Image.Image:
    if len(images) == 0:
        raise ValueError("images cannot be empty")
    gap = 10
    pad = 10
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * max(0, len(images) - 1)
    canvas = Image.new(
        "RGB",
        (width + pad * 2, height + pad * 2),
        color=bg_color,
    )
    y = pad
    for img in images:
        x = pad + (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas


def _write_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 2,
) -> None:
    if len(image_paths) == 0:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    cols = max(1, int(columns))
    rows = (len(images) + cols - 1) // cols
    gap = 8
    sheet = Image.new(
        "RGB",
        (cols * max_w + (cols + 1) * gap, rows * max_h + (rows + 1) * gap),
        color=(245, 245, 245),
    )
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = gap + col * (max_w + gap)
        y = gap + row * (max_h + gap)
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    for img in images:
        img.close()


def _pick_dataset_indices(
    dataset_len: int,
    *,
    num_samples: int,
    sampling: str,
    explicit_indices: list[int] | None,
) -> list[int]:
    if dataset_len <= 0:
        return []
    if explicit_indices:
        indices = sorted({int(i) for i in explicit_indices if 0 <= int(i) < dataset_len})
        if len(indices) == 0:
            raise ValueError("sample_indices were provided but none are within dataset bounds")
        return indices

    count = min(max(1, int(num_samples)), dataset_len)
    if sampling == "first":
        return list(range(count))
    if sampling == "random":
        return sorted(random.sample(range(dataset_len), k=count))
    if count == 1:
        return [0]
    return sorted({int(v) for v in np.linspace(0, dataset_len - 1, num=count, dtype=int).tolist()})


def _parse_override_value(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _set_nested(config: dict[str, Any], dotted_key: str, value) -> None:
    current = config
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError("empty override key")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _load_config(fname: Path, overrides: list[str]) -> dict[str, Any]:
    with fname.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML dict at {fname}, got {type(config)}")

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        _set_nested(config, key.strip(), _parse_override_value(raw_value.strip()))
    return config


def _resolve_branch_settings(
    args: dict[str, Any],
    branch: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[int, int], int]:
    cfg_data = dict(args.get("data", {}))
    cfg_mask = list(args.get("mask", []))
    crop_size = _to_hw_tuple(cfg_data.get("crop_size", 224), "data.crop_size")
    tubelet_size = int(cfg_data.get("tubelet_size", 2))

    if branch == "image":
        cfg_img_data = args.get("img_data", None)
        if not isinstance(cfg_img_data, dict) or not bool(cfg_img_data.get("enabled", False)):
            raise ValueError(
                "branch=image was requested, but img_data.enabled is not true in the config"
            )
        cfg_data = {**cfg_data, **dict(cfg_img_data)}
        crop_size = _to_hw_tuple(cfg_data.get("crop_size", crop_size), "img_data.crop_size")
        cfg_img_mask = args.get("img_mask", None)
        if cfg_img_mask is not None:
            cfg_mask = list(cfg_img_mask)
        if args.get("model", {}).get("img_temporal_dim_size", None) is not None:
            tubelet_size = 1

    return cfg_data, cfg_mask, crop_size, tubelet_size


def _build_dataset(
    *,
    cfg_data: dict[str, Any],
    cfg_data_aug: dict[str, Any],
    crop_size: tuple[int, int],
) -> EventVideoDataset:
    dataset_type = str(cfg_data.get("dataset_type", "eventdataset")).lower()
    if dataset_type not in {"eventdataset", "eventvoxel", "videodataset"}:
        raise ValueError(
            f"visualizer currently supports event datasets only, got dataset_type={dataset_type}"
        )

    dataset_paths = _ensure_list(cfg_data.get("datasets", []))
    if len(dataset_paths) == 0:
        raise ValueError("data.datasets must contain at least one path")
    dataset_fpcs = [int(v) for v in _ensure_list(cfg_data.get("dataset_fpcs", [8]))]
    if len(dataset_fpcs) == 0:
        raise ValueError("data.dataset_fpcs must contain at least one value")

    ar_range = tuple(
        float(v)
        for v in cfg_data_aug.get("random_resize_aspect_ratio", [0.75, 1.3333333333])
    )
    rr_scale = tuple(
        float(v) for v in cfg_data_aug.get("random_resize_scale", [0.3, 1.0])
    )
    random_horizontal_flip = bool(cfg_data_aug.get("random_horizontal_flip", True))
    interpolation = _resolve_interpolation(
        str(cfg_data_aug.get("interpolation", "bilinear"))
    )
    antialias = bool(cfg_data_aug.get("antialias", True))
    preserve_input_size = bool(cfg_data_aug.get("preserve_input_size", False))
    pad_to_hw_raw = cfg_data_aug.get("pad_to_hw", None)
    pad_to_hw = None
    if pad_to_hw_raw is not None:
        if not isinstance(pad_to_hw_raw, (list, tuple)) or len(pad_to_hw_raw) != 2:
            raise ValueError("data_aug.pad_to_hw must be [H, W] or null")
        pad_to_hw = (int(pad_to_hw_raw[0]), int(pad_to_hw_raw[1]))
    pad_value = float(cfg_data_aug.get("pad_value", 0.0))

    transform = make_event_transforms(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        crop_size=crop_size,
        interpolation=interpolation,
        antialias=antialias,
        apply_random_resized_crop=not preserve_input_size,
        pad_to_hw=pad_to_hw,
        pad_value=pad_value,
    )

    return EventVideoDataset(
        data_paths=dataset_paths,
        datasets_weights=cfg_data.get("datasets_weights", None),
        frames_per_clip=int(max(dataset_fpcs)),
        dataset_fpcs=dataset_fpcs,
        source_window_fpcs=cfg_data.get("source_window_fpcs", None),
        voxel_time_mode=cfg_data.get("voxel_time_mode", "channels"),
        voxel_temporal_bins=cfg_data.get("voxel_temporal_bins", None),
        frame_step=int(cfg_data.get("frame_sample_rate", 1)),
        fps=cfg_data.get("fps", None),
        num_clips=int(cfg_data.get("num_clips", 1)),
        transform=transform,
        shared_transform=None,
        random_clip_sampling=bool(cfg_data.get("random_clip_sampling", True)),
        allow_clip_overlap=bool(cfg_data.get("allow_clip_overlap", False)),
        file_pattern=str(cfg_data.get("file_pattern", "*.h5")),
        recursive=bool(cfg_data.get("recursive", True)),
        require_voxels_key=True,
        max_open_h5_files=int(cfg_data.get("max_open_h5_files", 32)),
        activity_filter_enabled=bool(cfg_data.get("activity_filter_enabled", False)),
        activity_filter_min_clip_mean_active_pixel_ratio=cfg_data.get(
            "activity_filter_min_clip_mean_active_pixel_ratio",
            None,
        ),
        activity_filter_min_clip_mean_activity_score=cfg_data.get(
            "activity_filter_min_clip_mean_activity_score",
            None,
        ),
        activity_filter_min_clip_active_window_ratio=cfg_data.get(
            "activity_filter_min_clip_active_window_ratio",
            None,
        ),
        activity_filter_active_window_threshold=cfg_data.get(
            "activity_filter_active_window_threshold",
            None,
        ),
    )


def _render_sample_sheet(
    *,
    sample_path: str,
    sample_index: int,
    draw_index: int,
    clip_tensor: torch.Tensor,
    clip_indices: list[int],
    mask_cfgs: list[dict[str, Any]],
    masks_enc: list[torch.Tensor],
    masks_pred: list[torch.Tensor],
    grid_h: int,
    grid_w: int,
    temporal_dim: int,
    expected_hw: tuple[int, int],
    frame_display_height: int,
    max_frames: int,
) -> tuple[Image.Image, list[str]]:
    clip_np = np.asarray(clip_tensor.detach().cpu(), dtype=np.float32)
    if clip_np.ndim != 4:
        raise ValueError(f"clip must be [C,T,H,W], got shape={clip_np.shape}")
    channels, num_frames, height, width = [int(v) for v in clip_np.shape]

    frame_ids = list(range(num_frames))
    if max_frames > 0 and len(frame_ids) > max_frames:
        frame_ids = sorted(
            {
                int(v)
                for v in np.linspace(0, len(frame_ids) - 1, num=max_frames, dtype=int).tolist()
            }
        )

    base_frames: list[Image.Image] = []
    for frame_idx in frame_ids:
        frame_rgb = _voxel_to_activity_rgb(clip_np[:, frame_idx])
        window_idx = clip_indices[frame_idx] if frame_idx < len(clip_indices) else -1
        label = f"t={frame_idx} win={window_idx}"
        base_frames.append(_annotate_panel(Image.fromarray(frame_rgb, mode="RGB"), label))

    summary_lines = [
        f"sample_index={sample_index} draw_index={draw_index}",
        f"path={sample_path}",
        f"clip_shape=[C={channels}, T={num_frames}, H={height}, W={width}]",
        f"clip_indices={clip_indices}",
        (
            f"clip_stats=min={float(np.nanmin(clip_np)):.6g} "
            f"max={float(np.nanmax(clip_np)):.6g} "
            f"mean={float(np.nanmean(clip_np)):.6g} "
            f"std={float(np.nanstd(clip_np)):.6g} "
            f"zero_ratio={float(np.mean(clip_np == 0.0)):.4f} "
            f"nonfinite={int(np.count_nonzero(~np.isfinite(clip_np)))}"
        ),
        "channel_abs_mean="
        + str([round(float(v), 6) for v in np.abs(clip_np).reshape(channels, -1).mean(axis=1)]),
        (
            f"mask_grid=[T={temporal_dim}, H={grid_h}, W={grid_w}] "
            f"expected_input_hw={expected_hw} actual_input_hw=({height}, {width})"
        ),
    ]

    rows = [
        _make_strip(
            title=(
                f"Activity | sample={sample_index} draw={draw_index} | "
                f"path={Path(sample_path).name}"
            ),
            frames=base_frames,
            frame_display_height=frame_display_height,
        )
    ]

    polarity_frames: list[Image.Image] = []
    for frame_idx in frame_ids:
        frame_rgb = _voxel_to_polarity_rgb(clip_np[:, frame_idx])
        window_idx = clip_indices[frame_idx] if frame_idx < len(clip_indices) else -1
        label = f"t={frame_idx} win={window_idx} red=pos blue=neg"
        polarity_frames.append(_annotate_panel(Image.fromarray(frame_rgb, mode="RGB"), label))
    rows.append(
        _make_strip(
            title=(
                f"Polarity | sample={sample_index} draw={draw_index} | "
                "red=positive blue=negative"
            ),
            frames=polarity_frames,
            frame_display_height=frame_display_height,
        )
    )

    for mask_id, (cfg_mask, enc_batch, pred_batch) in enumerate(
        zip(mask_cfgs, masks_enc, masks_pred)
    ):
        enc_indices = torch.as_tensor(enc_batch[0]).reshape(-1)
        pred_indices = torch.as_tensor(pred_batch[0]).reshape(-1)
        enc_volume = _indices_to_mask_volume(
            enc_indices,
            temporal_dim=temporal_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        )
        pred_volume = _indices_to_mask_volume(
            pred_indices,
            temporal_dim=temporal_dim,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        enc_ratio = float(enc_volume.mean())
        pred_ratio = float(pred_volume.mean())
        enc_t_ratio = [float(v) for v in enc_volume.reshape(temporal_dim, -1).mean(axis=1)]
        pred_t_ratio = [float(v) for v in pred_volume.reshape(temporal_dim, -1).mean(axis=1)]

        summary_lines.append(
            "mask[{mask_id}] cfg={cfg} enc_keep={enc_keep}/{total} ({enc_ratio:.2%}) "
            "pred={pred_keep}/{total} ({pred_ratio:.2%}) enc_t_ratio={enc_t_ratio} "
            "pred_t_ratio={pred_t_ratio}".format(
                mask_id=mask_id,
                cfg=cfg_mask,
                enc_keep=int(enc_volume.sum()),
                pred_keep=int(pred_volume.sum()),
                total=int(enc_volume.size),
                enc_ratio=enc_ratio,
                pred_ratio=pred_ratio,
                enc_t_ratio=[round(v, 4) for v in enc_t_ratio],
                pred_t_ratio=[round(v, 4) for v in pred_t_ratio],
            )
        )

        enc_frames: list[Image.Image] = []
        pred_frames: list[Image.Image] = []
        for frame_idx in frame_ids:
            patch_t = _frame_to_temporal_index(
                frame_idx,
                num_frames=num_frames,
                temporal_dim=temporal_dim,
            )
            base_rgb = _voxel_to_activity_rgb(clip_np[:, frame_idx])
            enc_overlay = _overlay_patch_mask(
                base_rgb,
                enc_volume[patch_t],
                color=(52, 168, 83),
            )
            pred_overlay = _overlay_patch_mask(
                base_rgb,
                pred_volume[patch_t],
                color=(217, 48, 37),
            )
            label = (
                f"t={frame_idx} patch_t={patch_t} "
                f"win={clip_indices[frame_idx] if frame_idx < len(clip_indices) else -1}"
            )
            enc_img = _draw_patch_grid(
                Image.fromarray(enc_overlay, mode="RGB"),
                grid_h=grid_h,
                grid_w=grid_w,
            )
            pred_img = _draw_patch_grid(
                Image.fromarray(pred_overlay, mode="RGB"),
                grid_h=grid_h,
                grid_w=grid_w,
            )
            enc_frames.append(_annotate_panel(enc_img, label))
            pred_frames.append(_annotate_panel(pred_img, label))

        rows.append(
            _make_strip(
                title=(
                    f"Mask {mask_id} Context | keep={int(enc_volume.sum())}/{int(enc_volume.size)} "
                    f"({enc_ratio:.2%}) | cfg={cfg_mask}"
                ),
                frames=enc_frames,
                frame_display_height=frame_display_height,
            )
        )
        rows.append(
            _make_strip(
                title=(
                    f"Mask {mask_id} Predictor | pred={int(pred_volume.sum())}/{int(pred_volume.size)} "
                    f"({pred_ratio:.2%}) | cfg={cfg_mask}"
                ),
                frames=pred_frames,
                frame_display_height=frame_display_height,
            )
        )

    return _stack_images(rows), summary_lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Visualize training-time event clips together with JEPA masks."
    )
    parser.add_argument(
        "--fname",
        type=Path,
        required=True,
        help="V-JEPA2.1 training YAML to visualize, e.g. configs/train_2_1/event/vitb16-h480w640-bins-t10.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mask_debug"),
        help="Directory where PNGs and the summary text file are written.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="How many dataset items to visualize when --sample-indices is not set.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit dataset indices to visualize.",
    )
    parser.add_argument(
        "--sampling",
        choices=["uniform", "random", "first"],
        default="uniform",
        help="How to choose dataset indices when --sample-indices is omitted.",
    )
    parser.add_argument(
        "--clip-id",
        type=int,
        default=0,
        help="Which clip to render when data.num_clips > 1. Training currently consumes clip 0.",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=1,
        help="How many times to resample the same dataset index (new clip sampling / new mask each draw).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help="Maximum number of frames to render per clip. Frames are subsampled uniformly when needed.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=160,
        help="Display height for each rendered frame tile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset sampling, augmentation, and mask generation.",
    )
    parser.add_argument(
        "--branch",
        choices=["video", "image"],
        default="video",
        help="Use the main training branch or the optional image branch config.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional config override. Can be repeated, e.g. --set data.datasets='[/path/train]'.",
    )
    parser.add_argument(
        "extra_overrides",
        nargs="*",
        help="Optional trailing key=value overrides for convenience.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = _parse_args()
    _seed_everything(int(cli_args.seed))

    fname = cli_args.fname
    if not fname.is_absolute():
        fname = PROJECT_ROOT / fname
    args = _load_config(fname, [*cli_args.overrides, *cli_args.extra_overrides])
    cfg_data, cfg_mask, crop_size, tubelet_size = _resolve_branch_settings(
        args,
        branch=str(cli_args.branch),
    )
    cfg_data_aug = dict(args.get("data_aug", {}))

    output_dir = cli_args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(args, sort_keys=False),
        encoding="utf-8",
    )

    dataset = _build_dataset(
        cfg_data=cfg_data,
        cfg_data_aug=cfg_data_aug,
        crop_size=crop_size,
    )
    dataset_fpcs = [int(v) for v in _ensure_list(cfg_data.get("dataset_fpcs", [8]))]
    patch_size = int(cfg_data.get("patch_size", 16))
    if crop_size[0] % patch_size != 0 or crop_size[1] % patch_size != 0:
        raise ValueError(
            f"crop_size={crop_size} must be divisible by patch_size={patch_size}"
        )
    grid_h = crop_size[0] // patch_size
    grid_w = crop_size[1] // patch_size

    mask_collator = MaskCollator(
        cfgs_mask=cfg_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=int(tubelet_size),
    )

    sample_indices = _pick_dataset_indices(
        len(dataset),
        num_samples=int(cli_args.num_samples),
        sampling=str(cli_args.sampling),
        explicit_indices=cli_args.sample_indices,
    )

    summary_lines = [
        f"branch={cli_args.branch}",
        f"fname={fname}",
        f"seed={cli_args.seed}",
        f"dataset_len={len(dataset)}",
        f"sample_indices={sample_indices}",
        f"crop_size={crop_size}",
        f"patch_size={patch_size}",
        f"tubelet_size={tubelet_size}",
        f"dataset_fpcs={dataset_fpcs}",
        f"source_window_fpcs={cfg_data.get('source_window_fpcs', None)}",
        f"voxel_time_mode={cfg_data.get('voxel_time_mode', 'channels')}",
        f"voxel_temporal_bins={cfg_data.get('voxel_temporal_bins', None)}",
        f"fps={cfg_data.get('fps', None)}",
        f"frame_sample_rate={cfg_data.get('frame_sample_rate', 1)}",
        "",
    ]
    written_images: list[Path] = []

    for sample_index in sample_indices:
        sample_path = dataset.samples[int(sample_index)]
        for draw_index in range(int(cli_args.num_draws)):
            loaded = dataset[int(sample_index)]
            collations = mask_collator([loaded])
            if len(collations) != 1:
                raise RuntimeError(
                    f"Expected exactly one collated fpc bucket for a single sample, got {len(collations)}"
                )

            udata, masks_enc, masks_pred = collations[0]
            num_clips = len(udata[0])
            if cli_args.clip_id < 0 or cli_args.clip_id >= num_clips:
                raise ValueError(
                    f"clip_id={cli_args.clip_id} is out of range for num_clips={num_clips}"
                )

            clip_tensor = torch.as_tensor(udata[0][cli_args.clip_id][0]).cpu()
            clip_indices = (
                torch.as_tensor(udata[2][cli_args.clip_id][0]).reshape(-1).cpu().tolist()
            )
            temporal_dim = max(1, int(clip_tensor.shape[1]) // int(tubelet_size))
            if temporal_dim <= 0:
                raise ValueError(
                    f"Failed to infer temporal mask dimension from clip_tensor.shape={tuple(clip_tensor.shape)} "
                    f"and tubelet_size={tubelet_size}"
                )

            sheet, sample_summary = _render_sample_sheet(
                sample_path=sample_path,
                sample_index=int(sample_index),
                draw_index=draw_index,
                clip_tensor=clip_tensor,
                clip_indices=clip_indices,
                mask_cfgs=cfg_mask,
                masks_enc=masks_enc,
                masks_pred=masks_pred,
                grid_h=grid_h,
                grid_w=grid_w,
                temporal_dim=temporal_dim,
                expected_hw=crop_size,
                frame_display_height=int(cli_args.frame_height),
                max_frames=int(cli_args.max_frames),
            )

            file_stem = Path(sample_path).stem.replace(" ", "_")
            out_path = output_dir / (
                f"sample_{int(sample_index):06d}_draw_{draw_index:02d}_{file_stem}.png"
            )
            sheet.save(out_path)
            written_images.append(out_path)
            summary_lines.extend(sample_summary)
            summary_lines.append(f"written={out_path}")
            summary_lines.append("")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    if len(written_images) > 1:
        _write_contact_sheet(
            written_images,
            output_dir / "contact_sheet.png",
            columns=min(2, len(written_images)),
        )

    print(f"Wrote {len(written_images)} visualization image(s) to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
