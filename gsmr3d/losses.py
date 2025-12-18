"""Loss utilities."""

from __future__ import annotations

import torch


def magnitude_tv_loss(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Isotropic total variation on the magnitude of a complex-valued volume."""

    magnitude = torch.sqrt(volume.real**2 + volume.imag**2 + eps)
    dz = magnitude[..., 1:, :, :] - magnitude[..., :-1, :, :]
    dy = magnitude[..., :, 1:, :] - magnitude[..., :, :-1, :]
    dx = magnitude[..., :, :, 1:] - magnitude[..., :, :, :-1]

    tv = dz.abs().mean() + dy.abs().mean() + dx.abs().mean()
    return tv
