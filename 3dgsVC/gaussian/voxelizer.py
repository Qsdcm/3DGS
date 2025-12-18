import torch
import torch.nn as nn
from typing import Tuple

class Voxelizer(nn.Module):
    def __init__(self, volume_shape: Tuple[int, int, int], device: str = "cuda:0"):
        super().__init__()
        self.volume_shape = volume_shape
        self.device = device
        # 不再预计算整个 grid_flat，太占显存且慢
        
    def forward(
        self,
        positions: torch.Tensor, # (N, 3) [-1, 1]
        scales: torch.Tensor,    # (N, 3)
        rotations: torch.Tensor, # (N, 4)
        density: torch.Tensor,   # (N,)
        chunk_size: int = 1000   # 每次处理1000个高斯
    ) -> torch.Tensor:
        D, H, W = self.volume_shape
        volume = torch.zeros((D, H, W), dtype=torch.complex64, device=self.device)
        
        N = positions.shape[0]
        
        # 预计算所有高斯的协方差逆
        from .gaussian_model import quaternion_to_rotation_matrix
        R = quaternion_to_rotation_matrix(rotations)
        inv_S_sq = torch.diag_embed(1.0 / (scales**2 + 1e-8))
        cov_inv = R @ inv_S_sq @ R.transpose(-1, -2) # (N, 3, 3)
        
        # 将位置转换为体素坐标
        pos_vox = (positions + 1) * torch.tensor([D/2, H/2, W/2], device=self.device) - 0.5
        
        # 3-sigma 截断
        cutoff = 3.0
        # 估算每个高斯的半径 (保守估计，取最大尺度的cutoff倍，转为体素单位)
        scale_vox = scales * torch.tensor([D/2, H/2, W/2], device=self.device)
        radius_vox = scale_vox.max(dim=-1)[0] * cutoff # (N,)
        
        # 预定义边界张量，修复 clamp 报错
        dims_tensor = torch.tensor([D, H, W], device=self.device)
        zeros_tensor = torch.zeros(3, device=self.device, dtype=torch.long)
        max_limits = dims_tensor - 1
        
        # 分块处理高斯点 (Splatting)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            
            # 当前块的数据
            p_chunk = pos_vox[i:end] # (B, 3)
            cov_inv_chunk = cov_inv[i:end] # (B, 3, 3)
            den_chunk = density[i:end] # (B,)
            r_chunk = radius_vox[i:end] # (B,)
            
            # 对块内的每个高斯，找到受影响的体素
            for b in range(end - i):
                p = p_chunk[b]
                r = r_chunk[b]
                ci = cov_inv_chunk[b] # (3, 3)
                den = den_chunk[b]
                
                # Bounding Box (修复了 clamp 参数类型)
                # 使用 min=Tensor, max=Tensor
                min_idx = torch.floor(p - r).long()
                min_idx = torch.clamp(min_idx, min=zeros_tensor, max=max_limits)
                
                max_idx_raw = torch.ceil(p + r).long()
                max_idx_clamped = torch.clamp(max_idx_raw, min=zeros_tensor, max=max_limits)
                max_idx = max_idx_clamped + 1
                
                # 检查有效性
                if (max_idx <= min_idx).any(): continue
                
                # 生成局部网格
                # 注意：arange 必须在循环内根据 min/max 生成
                d_rng = torch.arange(min_idx[0].item(), max_idx[0].item(), device=self.device)
                h_rng = torch.arange(min_idx[1].item(), max_idx[1].item(), device=self.device)
                w_rng = torch.arange(min_idx[2].item(), max_idx[2].item(), device=self.device)
                
                if d_rng.numel() == 0 or h_rng.numel() == 0 or w_rng.numel() == 0:
                    continue

                # Meshgrid
                zz, yy, xx = torch.meshgrid(d_rng, h_rng, w_rng, indexing='ij')
                grid_pts = torch.stack([zz, yy, xx], dim=-1).float() # (d, h, w, 3)
                
                # 偏移向量 (相对于中心，转换回归一化空间以匹配 cov_inv)
                # cov_inv 是基于归一化坐标计算的
                grid_norm = (grid_pts + 0.5) / torch.tensor([D/2, H/2, W/2], device=self.device) - 1.0
                p_norm = positions[i+b] # 取原始归一化坐标
                
                diff = grid_norm - p_norm # (d, h, w, 3)
                
                # 计算马氏距离 d^T Σ^-1 d
                # 优化: diff @ cov_inv -> temp; (temp * diff).sum(-1)
                diff_flat = diff.view(-1, 3)
                mahal = torch.sum((diff_flat @ ci) * diff_flat, dim=-1) # (pixels,)
                
                # 权重
                weights = torch.exp(-0.5 * mahal)
                
                # 累加到 volume
                weighted_den = weights * den
                
                # 使用 index slicing 累加
                volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += \
                    weighted_den.view(zz.shape)

        return volume