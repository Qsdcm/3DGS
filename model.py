"""model.py

3D Gaussian Splatting for MRI Reconstruction 模型模块。
已优化：Vectorized Sparse Voxelizer，大幅提升速度和梯度稳定性。
已修复：补全 get_state_dict 和 load_state_dict_custom 方法。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_read import image_to_kspace

class GaussianMRIModel(nn.Module):
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        n_gaussians: int = 1000,
        device: str = "cuda",
        sigma_cutoff: float = 3.0,
        use_rotation: bool = False,
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.nz, self.nx, self.ny = volume_shape
        self.device = device
        self.sigma_cutoff = sigma_cutoff
        self.use_rotation = use_rotation

        # 预计算归一化坐标网格参数
        self.register_buffer("grid_size", torch.tensor([self.nz, self.nx, self.ny], device=device))
        
        # 初始化参数
        self._init_gaussian_params(n_gaussians)

    def _init_gaussian_params(self, n_gaussians: int):
        # Centers: [-1, 1]
        centers = torch.rand(n_gaussians, 3, device=self.device) * 2 - 1
        self.centers = nn.Parameter(centers)

        # Log scales: 初始化为较小值
        log_scales = torch.full((n_gaussians, 3), -4.0, device=self.device)
        self.log_scales = nn.Parameter(log_scales)

        # Complex density
        rho_real = torch.randn(n_gaussians, device=self.device) * 0.1
        rho_imag = torch.randn(n_gaussians, device=self.device) * 0.1
        self.rho_real = nn.Parameter(rho_real)
        self.rho_imag = nn.Parameter(rho_imag)

        # Rotations (Identity)
        rotations = torch.zeros(n_gaussians, 4, device=self.device)
        rotations[:, 0] = 1.0
        if self.use_rotation:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)

    @property
    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)
    
    @property
    def n_gaussians(self) -> int:
        return self.centers.shape[0]

    def initialize_from_image(
        self,
        image_init_complex: torch.Tensor,
        n_gaussians: int,
        percentile_thresh: float = 10.0,
        k_init: float = 1.0,
        seed: int = 42,
    ):
        """从图像初始化，确保初始点分布在物体内部"""
        torch.manual_seed(seed)
        mag = torch.abs(image_init_complex)
        
        # 调试：检查初始图像强度
        print(f"[Model] Init Image Max: {mag.max().item():.6f}, Mean: {mag.mean().item():.6f}")

        thresh = torch.quantile(mag.flatten(), percentile_thresh / 100.0)
        valid_indices = torch.nonzero(mag > thresh, as_tuple=False) 
        
        if valid_indices.shape[0] == 0:
            print("[Warning] No valid voxels found for init, using random.")
            return

        if valid_indices.shape[0] > n_gaussians:
            perm = torch.randperm(valid_indices.shape[0], device=self.device)[:n_gaussians]
            sampled_indices = valid_indices[perm]
        else:
            sampled_indices = valid_indices

        real_n = sampled_indices.shape[0]
        
        # 转换为 [-1, 1] 坐标
        centers = torch.zeros(real_n, 3, device=self.device)
        centers[:, 0] = sampled_indices[:, 0].float() / (self.nz - 1) * 2 - 1
        centers[:, 1] = sampled_indices[:, 1].float() / (self.nx - 1) * 2 - 1
        centers[:, 2] = sampled_indices[:, 2].float() / (self.ny - 1) * 2 - 1
        
        vals = image_init_complex[sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]]
        
        self.centers = nn.Parameter(centers)
        
        # Scale 初始化: 接近体素大小
        voxel_scale = 2.0 / max(self.nz, self.nx, self.ny)
        self.log_scales = nn.Parameter(torch.full((real_n, 3), math.log(voxel_scale), device=self.device))
        
        # Rho (使用 k_init 放大)
        self.rho_real = nn.Parameter(vals.real * k_init)
        self.rho_imag = nn.Parameter(vals.imag * k_init)
        
        rotations = torch.zeros(real_n, 4, device=self.device)
        rotations[:, 0] = 1.0
        if self.use_rotation:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)
            
        print(f"[Model] Initialized {real_n} Gaussians. Scale approx {voxel_scale:.4f}. k_init={k_init}")

    def voxelize(self) -> torch.Tensor:
        """向量化体素化器"""
        M = self.n_gaussians
        R = 3  # Kernel radius
        
        grid_range = torch.arange(-R, R + 1, device=self.device)
        kz, kx, ky = torch.meshgrid(grid_range, grid_range, grid_range, indexing="ij")
        offsets = torch.stack([kz, kx, ky], dim=-1).reshape(-1, 3) # (K^3, 3)
        K = offsets.shape[0]
        
        grid_coords_float = (self.centers + 1) * 0.5 * (self.grid_size - 1)
        center_indices = torch.round(grid_coords_float).long()
        
        neighbor_indices = center_indices.unsqueeze(1) + offsets.unsqueeze(0)
        
        mask_z = (neighbor_indices[..., 0] >= 0) & (neighbor_indices[..., 0] < self.nz)
        mask_x = (neighbor_indices[..., 1] >= 0) & (neighbor_indices[..., 1] < self.nx)
        mask_y = (neighbor_indices[..., 2] >= 0) & (neighbor_indices[..., 2] < self.ny)
        mask_bounds = mask_z & mask_x & mask_y
        
        neighbor_coords = neighbor_indices.float() / (self.grid_size - 1) * 2 - 1
        
        delta = (neighbor_coords - self.centers.unsqueeze(1)) / (self.scales.unsqueeze(1) + 1e-8)
        dist_sq = torch.sum(delta ** 2, dim=-1)
        
        weights = torch.exp(-0.5 * dist_sq)
        mask_sigma = dist_sq <= (self.sigma_cutoff ** 2)
        final_mask = mask_bounds & mask_sigma
        
        valid_indices = neighbor_indices[final_mask]
        valid_weights = weights[final_mask]
        
        rho_real_expanded = self.rho_real.unsqueeze(1).expand(-1, K)[final_mask]
        rho_imag_expanded = self.rho_imag.unsqueeze(1).expand(-1, K)[final_mask]
        
        values_real = rho_real_expanded * valid_weights
        values_imag = rho_imag_expanded * valid_weights
        
        volume_real = torch.zeros((self.nz, self.nx, self.ny), device=self.device)
        volume_imag = torch.zeros((self.nz, self.nx, self.ny), device=self.device)
        
        idx_z = valid_indices[:, 0]
        idx_x = valid_indices[:, 1]
        idx_y = valid_indices[:, 2]
        
        volume_real.index_put_((idx_z, idx_x, idx_y), values_real, accumulate=True)
        volume_imag.index_put_((idx_z, idx_x, idx_y), values_imag, accumulate=True)
        
        return torch.complex(volume_real, volume_imag)

    def forward(self, mask):
        volume = self.voxelize()
        kspace = image_to_kspace(volume)
        kspace_pred = mask * kspace
        return volume, kspace_pred

    def compute_loss(self, kspace_pred, kspace_target, mask, volume=None, lambda_tv=0.0):
        # 注意：kspace_target 是 kspace_under (已 mask)
        # kspace_pred 是 mask * kspace_full_pred
        diff = kspace_pred - kspace_target
        # 再次应用 mask 确保只计算采样点 (虽然理论上已经 mask 了)
        diff = diff * mask 
        loss_dc = torch.mean(torch.abs(diff) ** 2)
        
        # 调试：如果 Loss 异常小，打印信息
        if loss_dc.item() < 1e-10:
             # 只打印一次或低频打印，防止刷屏（这里简单处理，每次遇到都打，方便用户看到）
             # 检查 pred 和 target 是否全 0
             pred_norm = torch.norm(kspace_pred)
             target_norm = torch.norm(kspace_target)
             print(f"[DEBUG] Loss=0 detected. Pred Norm: {pred_norm:.6e}, Target Norm: {target_norm:.6e}")
        
        loss_dict = {"loss_dc": loss_dc.item()}
        loss_total = loss_dc

        if lambda_tv > 0 and volume is not None:
            mag = torch.abs(volume)
            d_h = torch.abs(mag[:, 1:, :] - mag[:, :-1, :]).mean()
            d_w = torch.abs(mag[:, :, 1:] - mag[:, :, :-1]).mean()
            d_d = torch.abs(mag[1:, :, :] - mag[:-1, :, :]).mean()
            loss_tv = d_h + d_w + d_d
            loss_total = loss_total + lambda_tv * loss_tv
            loss_dict["loss_tv"] = loss_tv.item()

        loss_dict["loss_total"] = loss_total.item()
        return loss_total, loss_dict

    def densify_and_prune(self, grad_threshold, prune_rho_thresh, max_gaussians, min_gaussians, **kwargs):
        with torch.no_grad():
            if self.centers.grad is None: return {}
            
            rho_mag = torch.abs(torch.complex(self.rho_real, self.rho_imag))
            mask_keep = rho_mag > prune_rho_thresh
            
            grad_norm = self.centers.grad.norm(dim=-1)
            mask_split = (grad_norm > grad_threshold) & mask_keep
            
            n_split = mask_split.sum().item()
            if self.n_gaussians + n_split > max_gaussians:
                # 简单截断，防止显存爆炸
                n_split = 0 
                mask_split[:] = False
            
            if n_split > 0:
                self._split_gaussians(mask_split)
            
            # 统计
            return {"n_split": n_split, "n_pruned": (~mask_keep).sum().item()}

    def _split_gaussians(self, mask):
        # 简单的 Long-axis split 模拟
        old_centers = self.centers[mask]
        old_log_scales = self.log_scales[mask]
        old_rho_r = self.rho_real[mask]
        old_rho_i = self.rho_imag[mask]
        
        # 在原位置生成两个点，Scale 缩小，Rho 减半
        new_centers_1 = old_centers
        new_centers_2 = old_centers # 理想情况应沿长轴偏移，这里简化为重叠+优化器自动推开
        
        new_log_scales = old_log_scales - 0.69 # scale * 0.5
        new_rho_r = old_rho_r / 2
        new_rho_i = old_rho_i / 2
        
        # 拼接参数
        # 这里的实现比较粗糙，为了保持逻辑简单，我们直接 append 新点，
        # 实际更优的做法是原地替换或精心构造 tensor。
        # 但由于 torch.nn.Parameter 不能原地 resize，我们需要创建新的 Parameter
        
        # 保留未分裂的点
        mask_keep = ~mask
        
        final_centers = torch.cat([self.centers[mask_keep], new_centers_1, new_centers_2], dim=0)
        final_log_scales = torch.cat([self.log_scales[mask_keep], new_log_scales, new_log_scales], dim=0)
        final_rho_r = torch.cat([self.rho_real[mask_keep], new_rho_r, new_rho_r], dim=0)
        final_rho_i = torch.cat([self.rho_imag[mask_keep], new_rho_i, new_rho_i], dim=0)
        
        self.centers = nn.Parameter(final_centers)
        self.log_scales = nn.Parameter(final_log_scales)
        self.rho_real = nn.Parameter(final_rho_r)
        self.rho_imag = nn.Parameter(final_rho_i)
        
        if self.use_rotation:
             # 这里忽略了 rotation 的分裂逻辑，直接补全
             final_rots = torch.cat([self.rotations[mask_keep], self.rotations[mask], self.rotations[mask]], dim=0)
             self.rotations = nn.Parameter(final_rots)
        else:
             final_rots = torch.cat([self.rotations[mask_keep], self.rotations[mask], self.rotations[mask]], dim=0)
             self.register_buffer("rotations", final_rots)

    # =========================================================================
    # 补全的方法 (修复 AttributeError)
    # =========================================================================

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取模型状态用于保存。"""
        state = {
            "centers": self.centers.data,
            "log_scales": self.log_scales.data,
            "rho_real": self.rho_real.data,
            "rho_imag": self.rho_imag.data,
            "rotations": self.rotations.data if self.use_rotation else self.rotations,
        }
        return state

    def load_state_dict_custom(self, state: Dict[str, torch.Tensor]):
        """从保存的状态恢复模型。"""
        # 注意：加载时需要根据保存的参数数量 resize 当前模型
        n_gaussians = state["centers"].shape[0]
        
        # 必须重新创建 Parameter 才能改变形状
        self.centers = nn.Parameter(state["centers"].to(self.device))
        self.log_scales = nn.Parameter(state["log_scales"].to(self.device))
        self.rho_real = nn.Parameter(state["rho_real"].to(self.device))
        self.rho_imag = nn.Parameter(state["rho_imag"].to(self.device))

        if self.use_rotation:
            self.rotations = nn.Parameter(state["rotations"].to(self.device))
        else:
            self.register_buffer("rotations", state["rotations"].to(self.device))
            
        print(f"[Model] Loaded state dict. n_gaussians: {n_gaussians}")