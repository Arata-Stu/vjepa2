from __future__ import annotations

from typing import Mapping

import torch


class EventRepresentation:
    def convert(self, events: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class EventVoxelGrid(EventRepresentation):
    def __init__(
        self,
        input_size: tuple[int, int, int],
        normalize: bool,
        separate_polarity: bool = True,
        trilinear_interpolation: bool = True,
    ):
        if len(input_size) != 3:
            raise ValueError("input_size must be a tuple of (t_bins, height, width)")

        temporal_bins = int(input_size[0])
        if temporal_bins <= 0:
            raise ValueError("t_bins must be > 0")
        self.temporal_bins = temporal_bins
        self.separate_polarity = bool(separate_polarity)
        out_channels = temporal_bins * (2 if self.separate_polarity else 1)

        self.voxel_grid = torch.zeros(
            (out_channels, int(input_size[1]), int(input_size[2])),
            dtype=torch.float32,
            requires_grad=False,
        )
        self.nb_channels = out_channels
        self.normalize = bool(normalize)
        self.trilinear_interpolation = bool(trilinear_interpolation)

    def convert(self, events: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required_keys = ("x", "y", "p", "t")
        missing = [k for k in required_keys if k not in events]
        if missing:
            raise KeyError(f"missing event keys: {missing}")

        _, height, width = self.voxel_grid.shape
        if events["t"].numel() == 0:
            return self.voxel_grid.clone()

        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events["p"].device)
            voxel_grid = self.voxel_grid.clone()

            x = events["x"].float()
            y = events["y"].float()
            p = events["p"].float()
            t = events["t"].float()

            if t.numel() <= 1:
                t_norm = torch.zeros_like(t)
            else:
                duration = t[-1] - t[0]
                if torch.abs(duration).item() < 1e-12:
                    t_norm = torch.zeros_like(t)
                else:
                    t_norm = (self.temporal_bins - 1) * (t - t[0]) / duration

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()
            value = 2 * p - 1

            if self.trilinear_interpolation:
                for xlim in (x0, x0 + 1):
                    for ylim in (y0, y0 + 1):
                        for tlim in (t0, t0 + 1):
                            mask = (
                                (xlim < width)
                                & (xlim >= 0)
                                & (ylim < height)
                                & (ylim >= 0)
                                & (tlim >= 0)
                                & (tlim < self.temporal_bins)
                            )

                            interp_weights = (
                                (1 - (xlim - x).abs())
                                * (1 - (ylim - y).abs())
                                * (1 - (tlim - t_norm).abs())
                            )

                            if self.separate_polarity:
                                p_pos = p > 0
                                p_neg = ~p_pos

                                index_pos = (
                                    height * width * tlim.long()
                                    + width * ylim.long()
                                    + xlim.long()
                                )
                                tlim_neg = tlim.long() + self.temporal_bins
                                index_neg = (
                                    height * width * tlim_neg
                                    + width * ylim.long()
                                    + xlim.long()
                                )

                                mask_pos = mask & p_pos
                                mask_neg = mask & p_neg
                                voxel_grid.put_(
                                    index_pos[mask_pos],
                                    interp_weights[mask_pos],
                                    accumulate=True,
                                )
                                voxel_grid.put_(
                                    index_neg[mask_neg],
                                    interp_weights[mask_neg],
                                    accumulate=True,
                                )
                            else:
                                interp_signed = value * interp_weights
                                index = (
                                    height * width * tlim.long()
                                    + width * ylim.long()
                                    + xlim.long()
                                )
                                voxel_grid.put_(
                                    index[mask], interp_signed[mask], accumulate=True
                                )
            else:
                x_nn = x0
                y_nn = y0
                t_nn = torch.clamp(t0, min=0, max=self.temporal_bins - 1)
                mask = (
                    (x_nn < width)
                    & (x_nn >= 0)
                    & (y_nn < height)
                    & (y_nn >= 0)
                    & (t_nn >= 0)
                    & (t_nn < self.temporal_bins)
                )
                if self.separate_polarity:
                    p_pos = p > 0
                    p_neg = ~p_pos
                    index_pos = (
                        height * width * t_nn.long()
                        + width * y_nn.long()
                        + x_nn.long()
                    )
                    t_nn_neg = t_nn.long() + self.temporal_bins
                    index_neg = (
                        height * width * t_nn_neg
                        + width * y_nn.long()
                        + x_nn.long()
                    )
                    mask_pos = mask & p_pos
                    mask_neg = mask & p_neg
                    if mask_pos.any():
                        voxel_grid.put_(
                            index_pos[mask_pos],
                            torch.ones_like(index_pos[mask_pos], dtype=torch.float32),
                            accumulate=True,
                        )
                    if mask_neg.any():
                        voxel_grid.put_(
                            index_neg[mask_neg],
                            torch.ones_like(index_neg[mask_neg], dtype=torch.float32),
                            accumulate=True,
                        )
                else:
                    index = (
                        height * width * t_nn.long()
                        + width * y_nn.long()
                        + x_nn.long()
                    )
                    voxel_grid.put_(index[mask], value[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].numel() > 0:
                    values = voxel_grid[mask]
                    mean = values.mean()
                    std = values.std(unbiased=False)
                    if std > 0:
                        voxel_grid[mask] = (values - mean) / std
                    else:
                        voxel_grid[mask] = values - mean

        return voxel_grid


# Backward-compatible alias.
VoxelGrid = EventVoxelGrid
