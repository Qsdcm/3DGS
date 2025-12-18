"""Gaussian representation and adaptive control (Optimized)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn

def _to_tensor(data: torch.Tensor | Iterable[float], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32, device=device)

def quaternion_to_rotation_matrix_batch(q: torch.Tensor) -> torch.Tensor:
    """Batch convert normalized quaternions to rotation matrices (N, 3, 3)."""
    # q: (N, 4)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    w, x, y, z = q.unbind(dim=-1)

    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # Build matrix components
    row0 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row1 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)

def splat_gaussians(
    centers: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    rho: torch.Tensor,
    grid_shape: Tuple[int, int, int],
    *,
    radius_factor: float = 3.0,
) -> torch.Tensor:
    """
    Voxelize Gaussian points into a dense 3D complex grid.
    Optimized: Vectorized matrix computation and tight loop.
    """
    device = centers.device
    dtype = rho.dtype
    D, H, W = grid_shape
    volume = torch.zeros(grid_shape, dtype=dtype, device=device)

    # Pre-compute spatial dimensions
    span = torch.tensor([D - 1, H - 1, W - 1], dtype=torch.float32, device=device)
    
    # 1. Vectorized Pre-computation of Matrices (Critical for Speed)
    # R: (N, 3, 3)
    Rs = quaternion_to_rotation_matrix_batch(quats)
    
    # Sigma^-1 = R * S^-2 * R^T
    # InvScaleSq: (N, 3, 3) diagonal
    inv_scale_sq = torch.diag_embed(1.0 / (scales.clamp_min(1e-4) ** 2))
    
    # Precision Matrices: (N, 3, 3)
    # P = R @ inv_scale_sq @ R.T
    Ps = torch.bmm(Rs, torch.bmm(inv_scale_sq, Rs.transpose(1, 2)))

    # Convert centers to voxel coordinates: (N, 3)
    centers_voxel = centers * span
    
    # Get max radius per gaussian for bounding box (in voxels)
    max_scales = scales.max(dim=-1).values # (N,)
    radii = max_scales * radius_factor * span.max() # Approximation using max span dimension or per-axis
    # More precise per-axis radius would be better but this is safe upper bound
    # Let's use per-axis to be safe if scales are anisotropic
    radii_vec = scales * radius_factor * span # (N, 3)
    
    # Filter out small amplitudes to save time
    amp_mask = torch.abs(rho) > 1e-6
    indices = torch.nonzero(amp_mask, as_tuple=False).squeeze(-1)

    # 2. Loop over active gaussians (Still necessary in pure PyTorch for scatter, but lighter)
    # Ideally this would be a CUDA kernel.
    
    for idx in indices:
        idx = idx.item()
        
        # Bounding box
        c_vox = centers_voxel[idx]
        r_vox = radii_vec[idx]
        
        zmin = int(max(0, math.floor(c_vox[0] - r_vox[0])))
        zmax = int(min(D - 1, math.ceil(c_vox[0] + r_vox[0])))
        ymin = int(max(0, math.floor(c_vox[1] - r_vox[1])))
        ymax = int(min(H - 1, math.ceil(c_vox[1] + r_vox[1])))
        xmin = int(max(0, math.floor(c_vox[2] - r_vox[2])))
        xmax = int(min(W - 1, math.ceil(c_vox[2] + r_vox[2])))

        if zmax < zmin or ymax < ymin or xmax < xmin:
            continue

        # Grid generation (local patch)
        # Using meshgrid on small tensors is fast
        zs = torch.arange(zmin, zmax + 1, device=device, dtype=torch.float32)
        ys = torch.arange(ymin, ymax + 1, device=device, dtype=torch.float32)
        xs = torch.arange(xmin, xmax + 1, device=device, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        
        # Coordinates relative to center
        # (Patch_D, Patch_H, Patch_W, 3)
        rel = torch.stack([zz, yy, xx], dim=-1) - c_vox
        
        # Compute Gaussian values
        # P[idx]: (3, 3)
        P = Ps[idx] 
        
        # Vectorized quadratic form: x^T * P * x
        # rel: (..., 3) -> (M, 3)
        rel_flat = rel.reshape(-1, 3)
        # (M, 3) @ (3, 3) -> (M, 3) * (M, 3) -> sum -> (M,)
        exponent = torch.sum(torch.mm(rel_flat, P) * rel_flat, dim=1)
        
        weights = torch.exp(-0.5 * exponent).reshape(rel.shape[:-1])
        
        # Accumulate
        patch = weights.to(dtype=dtype) * rho[idx]
        volume[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] += patch

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
        self.log_scales = nn.Parameter(torch.log(scales.clamp_min(1e-6))) # Safe log
        self.quats = nn.Parameter(quats)
        # Store rho as (N, 2) real numbers to optimize better
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
        num_points: int = 500, # Paper uses sparse init
        magnitude_quantile: float = 0.90,
        # scale_voxels removed, calculated dynamically
        amplitude_shrink: float = 0.8,
        max_points: int = 400_000,
        adaptive: AdaptiveConfig | None = None,
        seed: int | None = None,
    ) -> "GaussianMRIModel":
        """Sample initialization points from the iFFT volume following paper details."""

        if volume.ndim != 3:
            raise ValueError("Expected a 3D complex volume.")

        device = volume.device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        # 1. Remove low intensity points (Quantile filtering)
        magnitude = volume.abs()
        flat = magnitude.reshape(-1)
        quantile_val = torch.quantile(flat, magnitude_quantile)
        
        candidates = torch.nonzero(flat >= quantile_val, as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            candidates = torch.arange(flat.numel(), device=device)

        # 2. Randomly sample M grid points
        if candidates.numel() > num_points:
            perm = torch.randperm(candidates.numel(), generator=rng, device=device)[:num_points]
            chosen = candidates[perm]
        else:
            chosen = candidates
            num_points = chosen.numel() # Update if fewer points found

        # Convert indices to normalized coordinates
        grid_shape = volume.shape
        chosen = chosen.long()
        depth, height, width = grid_shape
        yz = height * width
        z = torch.div(chosen, yz, rounding_mode="floor")
        rem = chosen - z * yz
        y = torch.div(rem, width, rounding_mode="floor")
        x = rem - y * width
        coords = torch.stack([z, y, x], dim=-1).float().to(device)
        
        # 3. Calculate Scale: average distance to nearest 3 grid points
        # Since these are grid points, calculating KNN exactly is expensive O(N^2) but feasible for N=500-2000
        # For grid points, "nearest 3" might just be adjacent voxels (dist=1), but let's calculate it to be generic.
        if num_points > 3:
            # Use cdists for pairwise distance
            dists = torch.cdist(coords, coords) # (N, N)
            # Mask diagonal (dist to self is 0)
            dists.fill_diagonal_(float('inf'))
            # Get 3 nearest neighbors
            nearest_dists, _ = dists.topk(3, largest=False, dim=1) # (N, 3)
            avg_dist = nearest_dists.mean(dim=1) # (N,)
            
            # Use isotropic scales initially based on this distance
            # Convert to normalized scale (since model uses normalized coords 0-1)
            # But the model parameters 'scales' are multiplied by 'span' in splatting, 
            # so 'scales' should be fraction of volume size. 
            # WAIT: splat_gaussians uses: center_voxel = center * span. cov_scales = scales[idx].
            # So scales should be in VOXEL units if splat uses them directly with voxel coords.
            # Looking at splat_gaussians logic:
            # rotation @ _inverse_diag_scales(cov_scales) -> precision matrix in voxel units.
            # So yes, scales are in VOXELS.
            
            # avg_dist is in voxels.
            scales_init = avg_dist.unsqueeze(1).repeat(1, 3) # (N, 3)
        else:
            # Fallback if too few points
            scales_init = torch.tensor([2.0, 2.0, 2.0], device=device).repeat(num_points, 1)

        # Normalize centers to [0, 1]
        centers = coords / torch.tensor([s - 1 for s in grid_shape], device=device)

        # 4. Rotation = 0 (Identity quaternion)
        quats = torch.zeros((num_points, 4), device=device)
        quats[:, -1] = 1.0 # w=1, x,y,z=0

        # Initialize Density
        rho = volume.reshape(-1)[chosen] * amplitude_shrink

        print(f"Initialized {num_points} Gaussians. Mean scale (voxels): {scales_init.mean().item():.2f}")

        return cls(
            grid_shape=grid_shape,
            centers=centers,
            scales=scales_init,
            quats=quats,
            rho=rho,
            max_points=max_points,
            adaptive=adaptive,
        )

    def prune(self, amplitude_threshold: float, gradient_threshold: float, grad_norms: torch.Tensor | None) -> int:
        # Combined pruning strategy
        amp = torch.sqrt(self.rho[..., 0] ** 2 + self.rho[..., 1] ** 2)
        
        # Keep if amplitude is high enough OR gradient is high enough (active region)
        # Paper says: "Gaussian points that the magnitude ... are too small while with large loss gradients are removed"
        # Wait, usually high gradient means "under-reconstructed", we want to KEEP or SPLIT.
        # Pruning usually removes points that contribute nothing.
        
        keep = amp >= amplitude_threshold
        
        # Also remove if scale is too huge (covering whole volume)? 
        # Optional: max_scale_limit = 0.5 * min(grid_shape)
        
        removed = self.num_points - int(keep.sum().item())
        if removed > 0:
            self._apply_mask(keep)
        return removed

    def long_axis_split(self, grad_norms: torch.Tensor, max_new: int) -> int:
        if grad_norms is None or grad_norms.numel() == 0:
            return 0

        # Identify candidates: High gradient
        mask = grad_norms > self.adaptive.gradient_threshold
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            return 0

        # Check capacity
        capacity = self.max_points - self.num_points
        if capacity <= 0:
            return 0
        if idxs.numel() > capacity:
            _, top_grad_idx = torch.topk(grad_norms[idxs], capacity)
            idxs = idxs[top_grad_idx]

        # Limit per step
        if idxs.numel() > max_new:
            _, top_grad_idx = torch.topk(grad_norms[idxs], max_new)
            idxs = idxs[top_grad_idx]

        # Fetch parent attributes
        centers = self.centers.detach()[idxs]
        scales = self.positive_scales().detach()[idxs]
        quats = self.normalized_quats().detach()[idxs]
        rho = self.complex_rho().detach()[idxs]

        new_centers = []
        new_scales = []
        new_quats = []
        new_rho = []

        span = self.span # (D, H, W)

        for c, s, q, r in zip(centers, scales, quats, rho):
            # Find longest axis
            axis = int(torch.argmax(s).item())
            
            # Paper: Split along longest axis, NO overlap.
            # Shift distance needs to be roughly 1 standard deviation or more.
            # s[axis] is essentially the sigma (in voxels).
            # We need to shift in NORMALIZED coordinates.
            
            sigma_voxel = s[axis]
            shift_voxel = sigma_voxel * 0.8 # Shift by 0.8 sigma -> distance 1.6 sigma. Enough to separate cores.
            
            span_axis = span[axis].item()
            shift_norm = float(shift_voxel / max(span_axis, 1.0))

            # Create two new centers
            c_neg = c.clone()
            c_pos = c.clone()
            c_neg[axis] -= shift_norm
            c_pos[axis] += shift_norm
            
            # Clamp to volume
            c_neg.clamp_(0.0, 1.0)
            c_pos.clamp_(0.0, 1.0)

            # Scales
            # Paper: "other two axes scaled by factor 0.85", long axis by "0.6"? (User prompt mentioned this)
            # Typically splitting reduces volume.
            scale_new = s.clone()
            scale_new[axis] *= 0.6  # Reduce length along split axis
            
            # Reduce width of other axes slightly to refine detail
            indices = [0, 1, 2]
            indices.remove(axis)
            for ax in indices:
                scale_new[ax] *= 0.85

            new_centers.extend([c_neg, c_pos])
            new_scales.extend([scale_new, scale_new])
            new_quats.extend([q, q])
            new_rho.extend([r * 0.5, r * 0.5]) # Conserve energy (roughly)

        # Update model
        # Remove parents (Split = Replace 1 with 2)
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
        # Ensure new scales are logged safely
        self.log_scales = nn.Parameter(torch.cat([self.log_scales, torch.log(scales.clamp_min(1e-6))], dim=0))
        self.quats = nn.Parameter(torch.cat([self.quats, quats], dim=0))
        rho_stack = torch.stack([rho.real, rho.imag], dim=-1)
        self.rho = nn.Parameter(torch.cat([self.rho, rho_stack], dim=0))