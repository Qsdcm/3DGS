import torch
import torch.nn as nn
from typing import Tuple
import math
 
class Voxelizer(nn.Module):
    def __init__(self, volume_shape: Tuple[int, int, int], device: str = "cuda:0"):
        super().__init__()
        self.volume_shape = volume_shape
        self.device = device
        
    def forward(
        self,
        positions: torch.Tensor, # (N, 3)
        scales: torch.Tensor,    # (N, 3)
        rotations: torch.Tensor, # (N, 4)
        density: torch.Tensor,   # (N,)
        chunk_size: int = 4096   # 优化: 增大 chunk_size 以利用 GPU 并行能力
    ) -> torch.Tensor:
        D, H, W = self.volume_shape
        device = self.device
        
        # 1. 预计算所有参数
        from .gaussian_model import quaternion_to_rotation_matrix
        
        # 转换到体素坐标系
        shape_tensor = torch.tensor([D, H, W], device=device).float()
        center_vox = (positions + 1.0) * 0.5 * shape_tensor - 0.5 
        scale_vox = scales * shape_tensor * 0.5
        
        # 计算 3-sigma 半径
        radius_vox = scale_vox.max(dim=-1)[0] * 3.0
        
        # 2. 【核心优化】智能排序：按半径从小到大排序
        # 这使得同一个 chunk 内的点大小一致，避免了"大点带小点"造成的显存浪费
        sort_indices = torch.argsort(radius_vox)
        
        # 对所有属性应用排序 (Autograd 会自动记录这一步，不影响梯度回传)
        center_sorted = center_vox[sort_indices]
        radius_sorted = radius_vox[sort_indices]
        density_sorted = density[sort_indices]
        scales_sorted = scales[sort_indices]
        rotations_sorted = rotations[sort_indices]
        
        # 计算协方差 (使用排序后的参数)
        R = quaternion_to_rotation_matrix(rotations_sorted)
        S_inv = torch.reciprocal(scales_sorted + 1e-8)
        L = R * S_inv.unsqueeze(1)
        cov_inv_sorted = L @ L.transpose(1, 2) # (N, 3, 3)


        # 初始化空白 Volume
        volume_flat = torch.zeros(D * H * W, dtype=torch.complex64, device=device)
        
        N = positions.shape[0]
        
        # 3. 分块并行处理 (Vectorized Splatting)
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            
            # 获取当前块 (因为已排序，这些点的半径都差不多大)
            c_center = center_sorted[i:end]
            c_cov_inv = cov_inv_sorted[i:end]
            c_density = density_sorted[i:end]
            c_radius = radius_sorted[i:end]
            
            # 动态确定 Kernel Size
            max_r = torch.ceil(c_radius.max()).int().item()
            max_r = min(max_r, 20) # 硬件上限保护
            
            # 生成局部网格
            k_rng = torch.arange(-max_r, max_r + 1, device=device)
            kz, ky, kx = torch.meshgrid(k_rng, k_rng, k_rng, indexing='ij')
            kernel_offsets = torch.stack([kz, ky, kx], dim=-1).reshape(-1, 3).float()
            
            # 扩展为 Batch: (B, K^3, 3)
            c_center_round = torch.round(c_center)
            global_coords = c_center_round.unsqueeze(1) + kernel_offsets.unsqueeze(0)
            
            # 筛选有效坐标
            gz, gy, gx = global_coords[..., 0], global_coords[..., 1], global_coords[..., 2]
            valid_mask = (gz >= 0) & (gz < D) & (gy >= 0) & (gy < H) & (gx >= 0) & (gx < W)
            
            # 计算马氏距离
            diff_vox = global_coords - c_center.unsqueeze(1)
            # 归一化距离用于协方差计算
            diff_norm = diff_vox / (shape_tensor * 0.5)
            
            diff_emb = diff_norm @ c_cov_inv
            mahal = (diff_emb * diff_norm).sum(dim=-1)
            
            # Mask: 3-sigma
            mask_final = valid_mask & (mahal <= 9.0)
            
            # 稀疏累加 (只处理有效值)
            valid_indices = torch.nonzero(mask_final, as_tuple=True)
            if valid_indices[0].numel() == 0: continue
            
            b_idx, p_idx = valid_indices
            
            # 计算权重
            weights = torch.exp(-0.5 * mahal[b_idx, p_idx])
            val_to_add = weights * c_density[b_idx]
            
            # 写入全局 Volume
            z_idx = gz[b_idx, p_idx].long()
            y_idx = gy[b_idx, p_idx].long()
            x_idx = gx[b_idx, p_idx].long()
            flat_indices = z_idx * (H * W) + y_idx * W + x_idx
            
            volume_flat.scatter_add_(0, flat_indices, val_to_add)
            
        return volume_flat.view(D, H, W)