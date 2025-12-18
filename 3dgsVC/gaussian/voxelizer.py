"""
Voxelizer: 将3D Gaussian表示转换为体素网格

论文公式(3): x_j = sum_i G_i^3(j | ρ_i, p_i, Σ_i)

对于每个体素位置j，计算所有Gaussian的贡献之和
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Tuple


class Voxelizer(nn.Module):
    """
    将3D Gaussian表示渲染为体素网格
    
    基于论文中的3D Gaussian voxelization方法
    """
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        device: str = "cuda:0"
    ):
        """
        Args:
            volume_shape: 体积形状 (D, H, W)
            device: 计算设备
        """
        super().__init__()
        
        self.volume_shape = volume_shape
        self.device = device
        
        # 预计算网格坐标
        self._create_grid()
        
    def _create_grid(self):
        """创建体素网格坐标"""
        D, H, W = self.volume_shape
        
        # 创建归一化坐标网格 [-1, 1]
        z = torch.linspace(-1, 1, D, device=self.device)
        y = torch.linspace(-1, 1, H, device=self.device)
        x = torch.linspace(-1, 1, W, device=self.device)
        
        # 创建网格
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        
        # (D, H, W, 3)
        self.grid = torch.stack([zz, yy, xx], dim=-1)
        
        # 展平为 (D*H*W, 3) 方便计算
        self.grid_flat = self.grid.reshape(-1, 3)
        
    def _compute_batch_contribution(
        self,
        positions: torch.Tensor,
        cov_inv: torch.Tensor,
        density: torch.Tensor,
        grid_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一个批次的体素贡献
        用于checkpointing以节省显存
        """
        B = grid_batch.shape[0]
        N = positions.shape[0]
        
        # 计算每个网格点到每个Gaussian中心的距离
        # diff: (B, N, 3)
        diff = grid_batch.unsqueeze(1) - positions.unsqueeze(0)
        
        # 计算马氏距离的平方: d^2 = (j-p)^T Σ^{-1} (j-p)
        # 使用einsum，因为在checkpoint内部，显存不是问题（只存当前批次）
        # mahal_sq[b, n] = sum_i sum_j diff[b,n,i] * cov_inv[n,i,j] * diff[b,n,j]
        mahal_sq = torch.einsum('bni,nij,bnj->bn', diff, cov_inv, diff)  # (B, N)
        
        # 高斯权重: exp(-0.5 * d^2)
        weights = torch.exp(-0.5 * mahal_sq)  # (B, N)
        
        # 加权求和: sum_i ρ_i * w_i
        # density: (N,) complex
        contribution = (weights * density.unsqueeze(0)).sum(dim=1)  # (B,) complex
        
        return contribution

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        density: torch.Tensor,
        batch_size: int = 100000  # 增加默认批处理大小，因为使用了checkpointing
    ) -> torch.Tensor:
        """
        渲染3D Gaussian到体素网格
        
        论文公式(3)和(4):
        x_j = sum_i ρ_i * exp(-1/2 * (j-p_i)^T Σ_i^{-1} (j-p_i))
        
        Args:
            positions: Gaussian中心位置 (N, 3), 归一化到[-1, 1]
            scales: Gaussian尺度 (N, 3), 非log空间
            rotations: 四元数旋转 (N, 4)
            density: 复数密度 (N,)
            batch_size: 批处理大小（用于控制显存）
            
        Returns:
            渲染的体素网格 (D, H, W), complex
        """
        N = positions.shape[0]
        D, H, W = self.volume_shape
        num_voxels = D * H * W
        
        # 根据高斯数量动态调整批处理大小
        # 使用checkpointing后，显存主要由checkpoint数量决定（每个checkpoint保存输入）
        # 输入很小，所以batch_size可以大一点以提高速度
        # 目标: 每批使用约 10GB 显存 (A6000有48GB)
        # 显存 ≈ batch_size * N * 20 bytes
        # batch_size ≈ 10GB / (N * 20) = 500MB / N
        max_batch = 500_000_000 // max(N, 1)
        batch_size = min(batch_size, max(50000, max_batch))
        
        # 构建协方差矩阵的逆
        # Σ = R S S^T R^T
        # Σ^{-1} = R S^{-1} S^{-T} R^T = R S^{-2} R^T
        from .gaussian_model import quaternion_to_rotation_matrix
        R = quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)
        
        # S^{-2} = diag(1/s1^2, 1/s2^2, 1/s3^2)
        inv_scales_sq = 1.0 / (scales ** 2 + 1e-8)  # (N, 3)
        inv_S_sq = torch.diag_embed(inv_scales_sq)  # (N, 3, 3)
        
        # Σ^{-1} = R @ S^{-2} @ R^T
        cov_inv = R @ inv_S_sq @ R.transpose(-1, -2)  # (N, 3, 3)
        
        # 释放不需要的中间变量
        del R, inv_scales_sq, inv_S_sq
        
        # 分批计算以节省显存
        volume = torch.zeros(num_voxels, dtype=torch.complex64, device=self.device)
        
        for start in range(0, num_voxels, batch_size):
            end = min(start + batch_size, num_voxels)
            
            # 当前批次的网格点
            grid_batch = self.grid_flat[start:end]  # (B, 3)
            
            # 使用checkpointing来节省显存
            # 这样反向传播时会重新计算前向传播，而不是保存中间激活
            if self.training and positions.requires_grad:
                contribution = torch.utils.checkpoint.checkpoint(
                    self._compute_batch_contribution,
                    positions,
                    cov_inv,
                    density,
                    grid_batch
                )
            else:
                contribution = self._compute_batch_contribution(
                    positions,
                    cov_inv,
                    density,
                    grid_batch
                )
            
            volume[start:end] = contribution
        
        # 释放 cov_inv
        del cov_inv
        
        # Reshape回体积形状
        volume = volume.reshape(D, H, W)
        
        return volume
        
        # 释放 cov_inv
        del cov_inv
        
        # Reshape回体积形状
        volume = volume.reshape(D, H, W)
        
        return volume
    
    def forward_efficient(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        density: torch.Tensor,
        cutoff_sigma: float = 3.0
    ) -> torch.Tensor:
        """
        高效版本：只计算Gaussian附近的体素
        
        使用截断高斯，只在中心周围cutoff_sigma个标准差内计算
        
        Args:
            positions: Gaussian中心位置 (N, 3)
            scales: Gaussian尺度 (N, 3)
            rotations: 四元数旋转 (N, 4)
            density: 复数密度 (N,)
            cutoff_sigma: 截断距离（标准差倍数）
            
        Returns:
            渲染的体素网格 (D, H, W), complex
        """
        N = positions.shape[0]
        D, H, W = self.volume_shape
        
        # 初始化输出
        volume = torch.zeros(D, H, W, dtype=torch.complex64, device=self.device)
        
        # 构建协方差矩阵的逆
        from .gaussian_model import quaternion_to_rotation_matrix
        R = quaternion_to_rotation_matrix(rotations)
        inv_scales_sq = 1.0 / (scales ** 2 + 1e-8)
        inv_S_sq = torch.diag_embed(inv_scales_sq)
        cov_inv = R @ inv_S_sq @ R.transpose(-1, -2)
        
        # 对每个Gaussian计算贡献
        for i in range(N):
            pos = positions[i]  # (3,)
            scale = scales[i]   # (3,)
            cov_inv_i = cov_inv[i]  # (3, 3)
            dens = density[i]   # complex scalar
            
            # 计算该Gaussian影响的范围（基于最大尺度）
            max_scale = scale.max().item()
            radius = cutoff_sigma * max_scale
            
            # 将归一化坐标转换为体素索引
            pos_voxel = (pos + 1) / 2 * torch.tensor([D, H, W], device=self.device, dtype=torch.float32)
            
            # 计算边界框
            min_idx = torch.floor(pos_voxel - radius * torch.tensor([D, H, W], device=self.device) / 2).long()
            max_idx = torch.ceil(pos_voxel + radius * torch.tensor([D, H, W], device=self.device) / 2).long()
            
            min_idx = torch.clamp(min_idx, min=0)
            max_idx = torch.clamp(max_idx, max=torch.tensor([D, H, W], device=self.device))
            
            # 提取局部网格
            local_grid = self.grid[
                min_idx[0]:max_idx[0],
                min_idx[1]:max_idx[1],
                min_idx[2]:max_idx[2]
            ]  # (d, h, w, 3)
            
            if local_grid.numel() == 0:
                continue
            
            # 计算到中心的距离
            diff = local_grid - pos  # (d, h, w, 3)
            
            # 计算马氏距离
            diff_flat = diff.reshape(-1, 3)  # (d*h*w, 3)
            mahal_sq = torch.einsum('bi,ij,bj->b', diff_flat, cov_inv_i, diff_flat)
            weights = torch.exp(-0.5 * mahal_sq).reshape(local_grid.shape[:-1])
            
            # 累加贡献
            volume[
                min_idx[0]:max_idx[0],
                min_idx[1]:max_idx[1],
                min_idx[2]:max_idx[2]
            ] += dens * weights
        
        return volume


def voxelize(
    gaussian_model,
    volume_shape: Tuple[int, int, int],
    device: str = "cuda:0",
    use_efficient: bool = False
) -> torch.Tensor:
    """
    便捷函数：将GaussianModel渲染为体素
    
    Args:
        gaussian_model: GaussianModel3D实例
        volume_shape: 体积形状
        device: 计算设备
        use_efficient: 是否使用高效版本
        
    Returns:
        体素网格 (D, H, W), complex
    """
    voxelizer = Voxelizer(volume_shape, device)
    
    params = gaussian_model.get_gaussian_params()
    
    if use_efficient:
        return voxelizer.forward_efficient(
            positions=params['positions'],
            scales=params['scales'],
            rotations=params['rotations'],
            density=params['density']
        )
    else:
        return voxelizer(
            positions=params['positions'],
            scales=params['scales'],
            rotations=params['rotations'],
            density=params['density']
        )
