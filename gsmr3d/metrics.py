"""Basic reconstruction metrics."""

from __future__ import annotations

import torch


def psnr3d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Peak signal-to-noise ratio on magnitude volumes."""

    diff = torch.abs(pred - target) ** 2
    mse = diff.mean().clamp_min(eps)
    peak = target.abs().max().clamp_min(eps)
    return float(20.0 * torch.log10(peak / torch.sqrt(mse)))


def ssim3d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """3D SSIM computed on magnitudes."""

    x = pred.abs().float().reshape(1, -1)
    y = target.abs().float().reshape(1, -1)

    mu_x = x.mean(dim=-1, keepdim=True)
    mu_y = y.mean(dim=-1, keepdim=True)
    sigma_x = ((x - mu_x) ** 2).mean(dim=-1, keepdim=True)
    sigma_y = ((y - mu_y) ** 2).mean(dim=-1, keepdim=True)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=-1, keepdim=True)

    l = target.abs().max().item()
    c1 = (0.01 * l) ** 2 + eps
    c2 = (0.03 * l) ** 2 + eps

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim.squeeze())
