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
        # scale 是归一化的 [-1, 1] 长度的一半? 
        # 通常 scale 对应 standard deviation。
        # 如果 positions 是 [-1, 1]，则 span 是 2。
        # scale_vox = scale * (Shape / 2)
        scale_vox = scales * torch.tensor([D/2, H/2, W/2], device=self.device)
        radius_vox = scale_vox.max(dim=-1)[0] * cutoff # (N,)
        
        # 分块处理高斯点 (Splatting)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            
            # 当前块的数据
            p_chunk = pos_vox[i:end] # (B, 3)
            cov_inv_chunk = cov_inv[i:end] # (B, 3, 3)
            den_chunk = density[i:end] # (B,)
            r_chunk = radius_vox[i:end] # (B,)
            
            # 对块内的每个高斯，找到受影响的体素
            # 这里如果不写 CUDA kernel，用 Python 循环遍历 B (1000) 是可接受的，比遍历 V (16M) 快得多
            # 为了进一步加速，可以尝试适度向量化，但变长边界框很难完全向量化
            
            for b in range(end - i):
                p = p_chunk[b]
                r = r_chunk[b]
                ci = cov_inv_chunk[b] # (3, 3)
                den = den_chunk[b]
                
                # Bounding Box
                min_idx = torch.floor(p - r).long().clamp(0, torch.tensor([D, H, W], device=self.device)-1)
                max_idx = torch.ceil(p + r).long().clamp(0, torch.tensor([D, H, W], device=self.device)-1) + 1
                
                # 检查有效性
                if (max_idx <= min_idx).any(): continue
                
                # 生成局部网格 (Vectorized within the box)
                # 显存优化: 如果box太大，可能需要截断或忽略
                d_rng = torch.arange(min_idx[0], max_idx[0], device=self.device)
                h_rng = torch.arange(min_idx[1], max_idx[1], device=self.device)
                w_rng = torch.arange(min_idx[2], max_idx[2], device=self.device)
                
                # Meshgrid
                zz, yy, xx = torch.meshgrid(d_rng, h_rng, w_rng, indexing='ij')
                grid_pts = torch.stack([zz, yy, xx], dim=-1).float() # (d, h, w, 3)
                
                # 偏移向量 (相对于中心)
                # 注意：cov_inv 是基于归一化坐标计算的，还是基于体素？
                # 之前的 gaussian_model 中 scale 是归一化的。
                # cov_inv 是归一化空间的。
                # grid_pts 是体素空间的。
                # 需要统一。最简单是将 grid_pts 转回归一化空间计算距离。
                
                grid_norm = (grid_pts + 0.5) / torch.tensor([D/2, H/2, W/2], device=self.device) - 1.0
                p_norm = positions[i+b] # 取原始归一化坐标
                
                diff = grid_norm - p_norm # (d, h, w, 3)
                
                # 计算马氏距离 d^T Σ^-1 d
                # (..., 3) @ (3, 3) @ (..., 3)^T
                # 优化: diff @ cov_inv -> temp; (temp * diff).sum(-1)
                diff_flat = diff.view(-1, 3)
                mahal = torch.sum((diff_flat @ ci) * diff_flat, dim=-1) # (pixels,)
                
                # 权重
                weights = torch.exp(-0.5 * mahal)
                
                # 累加到 volume
                # 只在 weights > exp(-cutoff^2/2) 时累加 (可选)
                
                weighted_den = weights * den
                
                # 使用 index_put_ (或者直接切片加，因为是在循环里)
                # volume[min:max] += values
                # 注意处理复数
                
                volume[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1], min_idx[2]:max_idx[2]] += \
                    weighted_den.view(zz.shape)

        return volume