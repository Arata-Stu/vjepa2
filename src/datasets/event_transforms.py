from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torchvision.transforms as tv_transforms
from torchvision.transforms import InterpolationMode


def _to_hw_tuple(crop_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(crop_size, int):
        return (crop_size, crop_size)
    if len(crop_size) != 2:
        raise ValueError("crop_size must be int or a 2-tuple/list")
    return int(crop_size[0]), int(crop_size[1])


def _to_optional_hw_tuple(hw: Sequence[int] | None) -> tuple[int, int] | None:
    if hw is None:
        return None
    if len(hw) != 2:
        raise ValueError("pad_to_hw must be a 2-tuple/list [H, W]")
    return int(hw[0]), int(hw[1])


class EventVideoTransform:
    """
    Basic event augmentation for JEPA stage-1.

    Input:  [T, H, W, C]
    Output: [C, T, H, W]
    """

    def __init__(
        self,
        *,
        random_horizontal_flip: bool = True,
        random_resize_aspect_ratio: tuple[float, float] = (3 / 4, 4 / 3),
        random_resize_scale: tuple[float, float] = (0.3, 1.0),
        crop_size: int | tuple[int, int] = 224,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
        apply_random_resized_crop: bool = True,
        pad_to_hw: tuple[int, int] | None = None,
        pad_value: float = 0.0,
    ):
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.crop_size = _to_hw_tuple(crop_size)
        self.interpolation = interpolation
        self.antialias = antialias
        self.apply_random_resized_crop = bool(apply_random_resized_crop)
        self.pad_to_hw = _to_optional_hw_tuple(pad_to_hw)
        self.pad_value = float(pad_value)

    def __call__(self, buffer: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(buffer):
            buffer = torch.as_tensor(buffer)
        buffer = buffer.to(torch.float32)
        if buffer.ndim != 4:
            raise ValueError(f"Expected buffer [T,H,W,C], got shape={tuple(buffer.shape)}")

        # [T,H,W,C] -> [T,C,H,W]
        buffer = buffer.permute(0, 3, 1, 2).contiguous()
        if self.apply_random_resized_crop:
            # Sample crop params from one frame and apply consistently across time.
            i, j, th, tw = tv_transforms.RandomResizedCrop.get_params(
                img=buffer[0],
                scale=self.random_resize_scale,
                ratio=self.random_resize_aspect_ratio,
            )
            cropped = buffer[:, :, i : i + th, j : j + tw]
            resized = F.interpolate(
                cropped,
                size=self.crop_size,
                mode=self.interpolation.value,
                align_corners=False
                if self.interpolation
                in (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)
                else None,
                antialias=self.antialias
                if self.interpolation
                in (InterpolationMode.BILINEAR, InterpolationMode.BICUBIC)
                else False,
            )
        else:
            # Keep native spatial resolution/aspect ratio.
            resized = buffer

        if self.random_horizontal_flip and torch.rand(1).item() < 0.5:
            resized = torch.flip(resized, dims=[3])  # Flip width.

        if self.pad_to_hw is not None:
            target_h, target_w = self.pad_to_hw
            _, _, cur_h, cur_w = resized.shape
            if cur_h > target_h or cur_w > target_w:
                raise ValueError(
                    f"pad_to_hw={self.pad_to_hw} is smaller than input HxW=({cur_h},{cur_w})"
                )
            pad_h = target_h - cur_h
            pad_w = target_w - cur_w
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                resized = F.pad(
                    resized,
                    pad=(pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=self.pad_value,
                )

        # [T,C,H,W] -> [C,T,H,W]
        return resized.permute(1, 0, 2, 3).contiguous()


def make_event_transforms(
    *,
    random_horizontal_flip: bool = True,
    random_resize_aspect_ratio: tuple[float, float] = (3 / 4, 4 / 3),
    random_resize_scale: tuple[float, float] = (0.3, 1.0),
    crop_size: int | tuple[int, int] = 224,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = True,
    apply_random_resized_crop: bool = True,
    pad_to_hw: tuple[int, int] | None = None,
    pad_value: float = 0.0,
) -> EventVideoTransform:
    return EventVideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        crop_size=crop_size,
        interpolation=interpolation,
        antialias=antialias,
        apply_random_resized_crop=apply_random_resized_crop,
        pad_to_hw=pad_to_hw,
        pad_value=pad_value,
    )
