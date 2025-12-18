import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import math

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """将四元数转换为旋转矩阵"""
    # 归一化四元数
    norm = quaternion.norm(dim=-1, keepdim=True)
    quaternion = quaternion / (norm + 1e-8)
    
    w, x, y, z = quaternion.unbind(-1)
    
    # 构建旋转矩阵
    row0 = torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1)
    row1 = torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1)
    row2 = torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
    
    return torch.stack([row0, row1, row2], dim=-2)

class GaussianModel3D(nn.Module):
    def __init__(
        self,
        num_points: int,
        volume_shape: Tuple[int, int, int],
        initial_positions: Optional[torch.Tensor] = None,
        initial_densities: Optional[torch.Tensor] = None,
        initial_scales: Optional[torch.Tensor] = None,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.num_points = num_points
        self.volume_shape = volume_shape
        self.device = device
        
        self._init_parameters(initial_positions, initial_densities, initial_scales)
        
    def _init_parameters(
        self,
        positions: Optional[torch.Tensor],
        densities: Optional[torch.Tensor],
        scales: Optional[torch.Tensor]
    ):
        N = self.num_points
        
        # 1. 位置
        if positions is None:
            positions = torch.rand(N, 3, device=self.device) * 2 - 1
        self.positions = nn.Parameter(positions)
        
        # 2. 尺度 (Log空间)
        if scales is None:
            # 默认给一个相对体素大小的尺度
            scales = torch.ones(N, 3, device=self.device) * (2.0 / self.volume_shape[0])
        # 存为log
        self.scales = nn.Parameter(torch.log(torch.clamp(scales, min=1e-8)))
        
        # 3. 旋转 (四元数 [1,0,0,0])
        rotations = torch.zeros(N, 4, device=self.device)
        rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(rotations)
        
        # 4. 密度
        if densities is None:
            densities = torch.randn(N, dtype=torch.complex64, device=self.device) * 0.1
            
        self.density_real = nn.Parameter(densities.real)
        self.density_imag = nn.Parameter(densities.imag)
        
    @property
    def density(self) -> torch.Tensor:
        return torch.complex(self.density_real, self.density_imag)
    
    def get_scale_values(self) -> torch.Tensor:
        return torch.exp(self.scales)

    # --- 修复部分：添加 trainer.py 需要的兼容方法 ---
    def get_scales(self) -> torch.Tensor:
        """兼容 trainer.py 的调用"""
        return self.get_scale_values()

    def get_densities(self) -> torch.Tensor:
        """兼容 trainer.py 的调用"""
        return self.density
    # -------------------------------------------
    
    def get_gaussian_params(self) -> Dict[str, torch.Tensor]:
        return {
            'positions': self.positions,
            'scales': self.get_scale_values(),
            'rotations': self.rotations,
            'density': self.density,
        }
        
    def get_optimizable_params(self, lr_position=1e-4, lr_density=1e-3, lr_scale=5e-4, lr_rotation=1e-4):
        return [
            {'params': [self.positions], 'lr': lr_position},
            {'params': [self.scales], 'lr': lr_scale},
            {'params': [self.rotations], 'lr': lr_rotation},
            {'params': [self.density_real], 'lr': lr_density},
            {'params': [self.density_imag], 'lr': lr_density},
        ]

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float = 0.0002,
        scale_threshold: float = 0.01,
        use_long_axis_splitting: bool = True
    ) -> int:
        """
        修正后的分裂逻辑：符合论文 Long-axis splitting
        """
        with torch.no_grad():
            scales = self.get_scale_values()
            max_scales = scales.max(dim=-1)[0]
            
            # 筛选条件：梯度大 且 尺度大
            mask = (grads > grad_threshold) & (max_scales > scale_threshold)
            if mask.sum() == 0: return 0
            
            # 提取父高斯
            p_pos = self.positions[mask]
            p_scale = scales[mask]
            p_rot = self.rotations[mask]
            p_den_r = self.density_real[mask]
            p_den_i = self.density_imag[mask]
            
            K = p_pos.shape[0]
            
            if use_long_axis_splitting:
                # 1. 找到最长轴
                longest_axis = p_scale.argmax(dim=-1) # (K,)
                
                # 2. 计算偏移 (无重叠)
                offset_val = p_scale[torch.arange(K), longest_axis] # (K,)
                
                # 局部偏移向量
                local_shift = torch.zeros_like(p_pos)
                local_shift[torch.arange(K), longest_axis] = offset_val * 1.0 # 1.0 sigma 偏移
                
                # 旋转到全局坐标系
                R = quaternion_to_rotation_matrix(p_rot) # (K, 3, 3)
                global_shift = torch.bmm(R, local_shift.unsqueeze(-1)).squeeze(-1) # (K, 3)
                
                new_pos_1 = p_pos + global_shift
                new_pos_2 = p_pos - global_shift
                
                # 3. 调整尺度
                new_scale = p_scale.clone()
                # 长轴变短 (* 0.6)
                new_scale[torch.arange(K), longest_axis] *= 0.6
                
                # 其他轴变窄 (* 0.85)
                mask_other = torch.ones_like(new_scale, dtype=torch.bool)
                mask_other[torch.arange(K), longest_axis] = False
                new_scale[mask_other] *= 0.85
                
                new_positions = torch.cat([new_pos_1, new_pos_2], dim=0)
                new_scales = torch.cat([new_scale, new_scale], dim=0)
                new_rotations = torch.cat([p_rot, p_rot], dim=0)
                # 能量守恒
                new_den_r = torch.cat([p_den_r/2, p_den_r/2], dim=0)
                new_den_i = torch.cat([p_den_i/2, p_den_i/2], dim=0)
                
            else:
                # 原始逻辑 Fallback
                std = p_scale
                offset = torch.randn_like(p_pos) * std * 0.5
                new_positions = torch.cat([p_pos - offset, p_pos + offset], dim=0)
                new_scales = torch.cat([p_scale/1.6, p_scale/1.6], dim=0)
                new_rotations = torch.cat([p_rot, p_rot], dim=0)
                new_den_r = torch.cat([p_den_r/2, p_den_r/2], dim=0)
                new_den_i = torch.cat([p_den_i/2, p_den_i/2], dim=0)

            # 更新参数
            keep_mask = ~mask
            self._update_params(keep_mask, new_positions, new_scales, new_rotations, new_den_r, new_den_i)
            return K

    def densify_and_clone(self, grads, grad_threshold, scale_threshold):
        with torch.no_grad():
             scales = self.get_scale_values()
             max_scales = scales.max(dim=-1)[0]
             mask = (grads > grad_threshold) & (max_scales <= scale_threshold)
             if mask.sum() == 0: return 0
             
             new_pos = self.positions[mask]
             new_scale = scales[mask]
             new_rot = self.rotations[mask]
             new_dr = self.density_real[mask]
             new_di = self.density_imag[mask]
             
             # Clone操作
             self.positions = nn.Parameter(torch.cat([self.positions, new_pos], dim=0))
             # 注意：new_scale 已经是 exp 后的值，需要重新 log
             self.scales = nn.Parameter(torch.cat([self.scales, torch.log(new_scale+1e-8)], dim=0))
             self.rotations = nn.Parameter(torch.cat([self.rotations, new_rot], dim=0))
             self.density_real = nn.Parameter(torch.cat([self.density_real, new_dr], dim=0))
             self.density_imag = nn.Parameter(torch.cat([self.density_imag, new_di], dim=0))
             self.num_points = self.positions.shape[0]
             return mask.sum().item()

    def prune(self, opacity_threshold):
        with torch.no_grad():
            density_mag = torch.abs(self.density)
            mask = density_mag > opacity_threshold
            if mask.sum() == self.num_points: return 0
            
            self._update_params(mask, None, None, None, None, None, is_prune=True)
            return (len(mask) - mask.sum().item())

    def _update_params(self, mask, new_pos=None, new_scale=None, new_rot=None, new_dr=None, new_di=None, is_prune=False):
        # 统一参数更新辅助函数
        if is_prune:
            self.positions = nn.Parameter(self.positions[mask])
            self.scales = nn.Parameter(self.scales[mask])
            self.rotations = nn.Parameter(self.rotations[mask])
            self.density_real = nn.Parameter(self.density_real[mask])
            self.density_imag = nn.Parameter(self.density_imag[mask])
        else:
            self.positions = nn.Parameter(torch.cat([self.positions[mask], new_pos], dim=0))
            self.scales = nn.Parameter(torch.cat([self.scales[mask], torch.log(new_scale + 1e-8)], dim=0))
            self.rotations = nn.Parameter(torch.cat([self.rotations[mask], new_rot], dim=0))
            self.density_real = nn.Parameter(torch.cat([self.density_real[mask], new_dr], dim=0))
            self.density_imag = nn.Parameter(torch.cat([self.density_imag[mask], new_di], dim=0))
            
        self.num_points = self.positions.shape[0]

    @classmethod
    def from_image(cls, image: torch.Tensor, num_points: int, initial_scale: float = 2.0, device: str = "cuda:0"):
        """符合论文的初始化"""
        D, H, W = image.shape
        # 1. 阈值过滤
        mag = torch.abs(image).flatten()
        threshold = torch.quantile(mag, 0.90)
        candidates = torch.nonzero(mag > threshold).squeeze()
        
        if candidates.numel() < num_points:
            candidates = torch.arange(mag.numel(), device=device)
        
        # 2. 随机采样
        indices_idx = torch.randperm(candidates.numel(), device=device)[:num_points]
        indices = candidates[indices_idx]
        
        # 坐标转换
        z = indices // (H * W)
        rem = indices % (H * W)
        y = rem // W
        x = rem % W
        
        positions = torch.stack([
            z.float() / D * 2 - 1,
            y.float() / H * 2 - 1,
            x.float() / W * 2 - 1
        ], dim=-1)
        
        # 3. 初始尺度 (KNN=3)
        if num_points > 3:
            dist_mat = torch.cdist(positions, positions)
            dist_mat.fill_diagonal_(float('inf'))
            vals, _ = dist_mat.topk(3, largest=False, dim=1)
            mean_dist = vals.mean(dim=1, keepdim=True)
            scales_init = mean_dist.repeat(1, 3)
        else:
            scales_init = torch.ones(num_points, 3, device=device) * (2.0/D)
            
        # 4. 密度初始化 (scale down)
        img_flat = image.reshape(-1)
        densities = img_flat[indices] * 0.15
        
        return cls(
            num_points=num_points,
            volume_shape=image.shape,
            initial_positions=positions,
            initial_densities=densities,
            initial_scales=scales_init,
            device=device
        )