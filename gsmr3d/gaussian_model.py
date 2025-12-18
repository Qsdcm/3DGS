"""Gaussian representation and adaptive control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


def _to_tensor(data: torch.Tensor | Iterable[float], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized quaternions to rotation matrices."""

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def _inverse_diag_scales(scales: torch.Tensor) -> torch.Tensor:
    return torch.diag_embed((1.0 / (scales.clamp_min(1e-4) ** 2)))


def splat_gaussians(
    centers: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    rho: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    *,
    radius_factor: float = 3.0,
) -> torch.Tensor:
    """Voxelize Gaussian points into a dense 3D complex grid."""

    device = centers.device
    dtype = rho.dtype
    volume = torch.zeros(grid_shape, dtype=dtype, device=device)

    span = torch.tensor([dim - 1 for dim in grid_shape], dtype=torch.float32, device=device)
    num_points = centers.shape[0]

    for idx in range(num_points):
        amplitude = rho[idx]
        if torch.abs(amplitude) < 1e-6:
            continue

        center = centers[idx]
        cov_scales = scales[idx]
        quat = quats[idx]
        center_voxel = center * span

        patch_radius = float(torch.max(cov_scales) * radius_factor)
        if patch_radius < 1e-3:
            continue

        zmin = max(0, int(math.floor(center_voxel[0].item() - patch_radius)))
        zmax = min(grid_shape[0] - 1, int(math.ceil(center_voxel[0].item() + patch_radius)))
        ymin = max(0, int(math.floor(center_voxel[1].item() - patch_radius)))
        ymax = min(grid_shape[1] - 1, int(math.ceil(center_voxel[1].item() + patch_radius)))
        xmin = max(0, int(math.floor(center_voxel[2].item() - patch_radius)))
        xmax = min(grid_shape[2] - 1, int(math.ceil(center_voxel[2].item() + patch_radius)))

        if zmax < zmin or ymax < ymin or xmax < xmin:
            continue

        zs = torch.arange(zmin, zmax + 1, device=device, dtype=torch.float32)
        ys = torch.arange(ymin, ymax + 1, device=device, dtype=torch.float32)
        xs = torch.arange(xmin, xmax + 1, device=device, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")

        coords = torch.stack([zz, yy, xx], dim=-1)
        rel = coords - center_voxel
        rel_flat = rel.reshape(-1, 3)

        rotation = quaternion_to_rotation_matrix(quat.unsqueeze(0))[0]
        precision = rotation @ _inverse_diag_scales(cov_scales.unsqueeze(0))[0] @ rotation.T

        exponent = torch.einsum("bi,ij,bj->b", rel_flat, precision, rel_flat)
        weights = torch.exp(-0.5 * exponent).reshape(rel.shape[:-1])

        patch = weights.to(dtype=dtype) * amplitude
        volume[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1] += patch

    return volume


@dataclass
class AdaptiveConfig:
    densify_interval: int = 100
    gradient_threshold: float = 0.15
    amplitude_threshold: float = 1e-4
    max_add_per_interval: int = 2048


class GaussianMRIModel(nn.Module):
    """MRI volume represented by a mixture of 3D Gaussians."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int],
        centers: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        rho: torch.Tensor,
        *,
        max_points: int = 400_000,
        adaptive: AdaptiveConfig | None = None,
    ) -> None:
        super().__init__()
        self.grid_shape = tuple(int(x) for x in grid_shape)
        self.max_points = int(max_points)
        self.adaptive = adaptive or AdaptiveConfig()

        device = centers.device
        self.register_buffer("span", torch.tensor([s - 1 for s in self.grid_shape], dtype=torch.float32, device=device))
        self.register_buffer("device_indicator", torch.tensor(0.0, device=device))

        self.centers = nn.Parameter(centers)
        self.log_scales = nn.Parameter(torch.log(scales.clamp_min(1e-3)))
        self.quats = nn.Parameter(quats)
        rho_stack = torch.stack([rho.real, rho.imag], dim=-1)
        self.rho = nn.Parameter(rho_stack)

    @property
    def num_points(self) -> int:
        return int(self.centers.shape[0])

    @property
    def device(self) -> torch.device:
        return self.device_indicator.device

    def positive_scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def complex_rho(self) -> torch.Tensor:
        return torch.complex(self.rho[..., 0], self.rho[..., 1])

    def normalized_quats(self) -> torch.Tensor:
        return self.quats / self.quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self) -> torch.Tensor:
        return splat_gaussians(
            centers=self.centers,
            scales=self.positive_scales(),
            quats=self.normalized_quats(),
            rho=self.complex_rho(),
            grid_shape=self.grid_shape,
        )

    @classmethod
    def from_initial_volume(
        cls,
        volume: torch.Tensor,
        *,
        num_points: int = 50_000,
        magnitude_quantile: float = 0.90,
        scale_voxels: Tuple[float, float, float] = (3.0, 3.0, 3.0),
        amplitude_shrink: float = 0.8,
        max_points: int = 400_000,
        adaptive: AdaptiveConfig | None = None,
        seed: int | None = None,
    ) -> "GaussianMRIModel":
        """Sample initialization points from the iFFT volume."""

        if volume.ndim != 3:
            raise ValueError("Expected a 3D complex volume.")

        device = volume.device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        magnitude = volume.abs()
        flat = magnitude.reshape(-1)
        quantile = torch.quantile(flat, magnitude_quantile)

        candidates = torch.nonzero(flat >= quantile, as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            candidates = torch.arange(flat.numel(), device=device)

        if candidates.numel() > num_points:
            perm = torch.randperm(candidates.numel(), generator=rng, device=device)[:num_points]
            chosen = candidates[perm]
        else:
            chosen = candidates

        grid_shape = volume.shape
        chosen = chosen.long()
        depth, height, width = grid_shape
        yz = height * width
        z = torch.div(chosen, yz, rounding_mode="floor")
        rem = chosen - z * yz
        y = torch.div(rem, width, rounding_mode="floor")
        x = rem - y * width
        coords = torch.stack([z, y, x], dim=-1).float().to(device)
        centers = coords / torch.tensor([s - 1 for s in grid_shape], device=device)

        scales = torch.tensor(scale_voxels, device=device, dtype=torch.float32).repeat(centers.shape[0], 1)
        quats = torch.zeros((centers.shape[0], 4), device=device)
        quats[:, -1] = 1.0

        rho = volume.reshape(-1)[chosen] * amplitude_shrink

        return cls(
            grid_shape=grid_shape,
            centers=centers,
            scales=scales,
            quats=quats,
            rho=rho,
            max_points=max_points,
            adaptive=adaptive,
        )

    def prune(self, amplitude_threshold: float, gradient_threshold: float, grad_norms: torch.Tensor | None) -> int:
        amp = torch.sqrt(self.rho[..., 0] ** 2 + self.rho[..., 1] ** 2)
        keep = amp >= amplitude_threshold
        if grad_norms is not None:
            keep |= grad_norms >= gradient_threshold

        removed = self.num_points - int(keep.sum().item())
        if removed > 0:
            self._apply_mask(keep)
        return removed

    def long_axis_split(self, grad_norms: torch.Tensor, max_new: int) -> int:
        if grad_norms is None or grad_norms.numel() == 0:
            return 0

        mask = grad_norms > self.adaptive.gradient_threshold
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            return 0

        capacity = self.max_points - self.num_points
        if capacity <= 0:
            return 0

        if idxs.numel() > capacity:
            idxs = idxs[:capacity]

        idxs = idxs[:max_new]
        if idxs.numel() == 0:
            return 0

        centers = self.centers.detach()[idxs]
        scales = self.positive_scales().detach()[idxs]
        quats = self.normalized_quats().detach()[idxs]
        rho = self.complex_rho().detach()[idxs]

        new_centers = []
        new_scales = []
        new_quats = []
        new_rho = []

        span = self.span

        for c, s, q, r in zip(centers, scales, quats, rho):
            axis = int(torch.argmax(s).item())
            shift = s[axis] * 0.5
            span_axis = span[axis].item() if span[axis] > 0 else 1.0
            shift_norm = float(shift / max(span_axis, 1.0))

            c_voxel = c.clone()
            c_voxel_norm = c_voxel.clone()
            offset = shift_norm

            neg = c_voxel_norm.clone()
            pos = c_voxel_norm.clone()
            neg[axis] = torch.clamp(neg[axis] - offset, 0.0, 1.0)
            pos[axis] = torch.clamp(pos[axis] + offset, 0.0, 1.0)

            scale_new = s.clone()
            scale_new[axis] *= 0.6
            for ax in range(3):
                if ax != axis:
                    scale_new[ax] *= 0.85

            new_centers.extend([neg, pos])
            new_scales.extend([scale_new.clone(), scale_new.clone()])
            new_quats.extend([q.clone(), q.clone()])
            new_rho.extend([r * 0.5, r * 0.5])

        keep_mask = torch.ones(self.num_points, dtype=torch.bool, device=self.device)
        keep_mask[idxs] = False
        self._apply_mask(keep_mask)

        if new_centers:
            self._append_gaussians(
                centers=torch.stack(new_centers, dim=0),
                scales=torch.stack(new_scales, dim=0),
                quats=torch.stack(new_quats, dim=0),
                rho=torch.stack(new_rho, dim=0),
            )

        return len(new_centers)

    def _apply_mask(self, mask: torch.Tensor) -> None:
        self.centers = nn.Parameter(self.centers[mask])
        self.log_scales = nn.Parameter(self.log_scales[mask])
        self.quats = nn.Parameter(self.quats[mask])
        self.rho = nn.Parameter(self.rho[mask])

    def _append_gaussians(
        self,
        *,
        centers: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        rho: torch.Tensor,
    ) -> None:
        if centers.numel() == 0:
            return
        self.centers = nn.Parameter(torch.cat([self.centers, centers], dim=0))
        self.log_scales = nn.Parameter(torch.cat([self.log_scales, torch.log(scales.clamp_min(1e-3))], dim=0))
        self.quats = nn.Parameter(torch.cat([self.quats, quats], dim=0))
        rho_stack = torch.stack([rho.real, rho.imag], dim=-1)
        self.rho = nn.Parameter(torch.cat([self.rho, rho_stack], dim=0))
