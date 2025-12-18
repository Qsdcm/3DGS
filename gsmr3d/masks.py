"""Sampling masks inspired by the paper."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import torch


def gaussian_undersampling_mask(
    shape: Iterable[int],
    acceleration: float,
    *,
    center_fraction: float = 0.08,
    sigma: Tuple[float, float, float] | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a 3D variable-density Gaussian sampling mask."""

    dims = tuple(int(s) for s in shape)
    if len(dims) != 3:
        raise ValueError("shape must be 3D")

    if acceleration <= 1.0:
        return torch.ones(dims, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    sigma = sigma or (0.35, 0.35, 0.35)
    total_voxels = int(np.prod(dims))
    target_samples = max(1, int(total_voxels / acceleration))

    mask = np.zeros(dims, dtype=np.float32)

    center_widths = [max(1, int(round(center_fraction * s))) for s in dims]
    center_slices = []
    for size, dim in zip(center_widths, dims):
        half = size // 2
        start = max(0, dim // 2 - half)
        end = min(dim, start + size)
        center_slices.append(slice(start, end))
    mask[tuple(center_slices)] = 1.0
    current_samples = int(mask.sum())

    need = max(0, target_samples - current_samples)
    if need == 0:
        tensor = torch.from_numpy(mask)
        return tensor.to(device=device)

    z = np.linspace(-1.0, 1.0, dims[0], dtype=np.float32)
    y = np.linspace(-1.0, 1.0, dims[1], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, dims[2], dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    gaussian = np.exp(
        -(
            (zz ** 2) / (2 * sigma[0] ** 2)
            + (yy ** 2) / (2 * sigma[1] ** 2)
            + (xx ** 2) / (2 * sigma[2] ** 2)
        )
    )

    flat_gaussian = gaussian.reshape(-1)
    flat_mask = mask.reshape(-1)
    candidates = np.where(flat_mask < 0.5)[0]
    weights = flat_gaussian[candidates]

    if weights.sum() == 0:
        weights = np.ones_like(weights, dtype=np.float32)
    weights = weights / weights.sum()

    choose = min(need, candidates.shape[0])
    selected = rng.choice(candidates, size=choose, replace=False, p=weights)
    flat_mask[selected] = 1.0

    tensor = torch.from_numpy(flat_mask.reshape(dims))
    return tensor.to(device=device)
