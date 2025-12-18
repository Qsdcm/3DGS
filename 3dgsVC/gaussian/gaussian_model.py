"""
3D Gaussian Model for MRI Reconstruction

实现论文中的3D Gaussian表示模型
每个体素由多个可学习的3D Gaussian点组合表示

论文公式(3): x_j = sum_i G_i^3(j | ρ_i, p_i, Σ_i)
论文公式(4): G_i^3 = ρ_i * exp(-1/2 * (j-p_i)^T Σ_i^{-1} (j-p_i))
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import math


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵
    
    论文中使用四元数参数化旋转矩阵R_i
    
    Args:
        quaternion: 四元数 (N, 4) [w, x, y, z]
        
    Returns:
        旋转矩阵 (N, 3, 3)
    """
    # 归一化四元数
    quaternion = quaternion / (quaternion.norm(dim=-1, keepdim=True) + 1e-8)
    
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    # 构建旋转矩阵
    R = torch.zeros(quaternion.shape[0], 3, 3, device=quaternion.device, dtype=quaternion.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    
    return R


def build_covariance_matrix(scale: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    构建协方差矩阵
    
    论文公式: Σ_i = R_i S_i S_i^T R_i^T
    
    Args:
        scale: 尺度参数 (N, 3)
        rotation: 四元数 (N, 4)
        
    Returns:
        协方差矩阵 (N, 3, 3)
    """
    # 获取旋转矩阵
    R = quaternion_to_rotation_matrix(rotation)  # (N, 3, 3)
    
    # 构建尺度矩阵 S = diag(s1, s2, s3)
    # 使用exp确保尺度为正
    S = torch.diag_embed(torch.exp(scale))  # (N, 3, 3)
    
    # Σ = R S S^T R^T = R S^2 R^T
    # S^2 = diag(s1^2, s2^2, s3^2)
    L = R @ S  # (N, 3, 3)
    cov = L @ L.transpose(-1, -2)  # (N, 3, 3)
    
    return cov


class GaussianModel3D(nn.Module):
    """
    3D Gaussian表示模型
    
    参数化:
    - position (p_i): 3D位置 (N, 3)
    - scale (s_i): 3D尺度 (N, 3), 使用log空间
    - rotation (q_i): 四元数旋转 (N, 4)
    - density_real (ρ_i): 复数密度的实部 (N,)
    - density_imag (ρ_i): 复数密度的虚部 (N,)
    
    MRI信号是复值的，因此density需要是复数
    """
    
    def __init__(
        self,
        num_points: int,
        volume_shape: Tuple[int, int, int],
        initial_positions: Optional[torch.Tensor] = None,
        initial_densities: Optional[torch.Tensor] = None,
        initial_scale: float = 2.0,
        device: str = "cuda:0"
    ):
        """
        Args:
            num_points: Gaussian点数量
            volume_shape: 体积形状 (D, H, W)
            initial_positions: 初始位置，如果为None则随机初始化
            initial_densities: 初始密度值（复数），如果为None则从zero-filled图像采样
            initial_scale: 初始尺度
            device: 计算设备
        """
        super().__init__()
        
        self.num_points = num_points
        self.volume_shape = volume_shape
        self.device = device
        self.initial_scale = initial_scale
        
        # 初始化参数
        self._init_parameters(initial_positions, initial_densities)
        
    def _init_parameters(
        self,
        initial_positions: Optional[torch.Tensor],
        initial_densities: Optional[torch.Tensor]
    ):
        """
        初始化Gaussian参数
        
        论文Section II-C: 
        使用iFFT重建的图像进行初始化
        """
        N = self.num_points
        D, H, W = self.volume_shape
        
        # 位置参数 (N, 3) - 归一化到 [-1, 1]
        if initial_positions is not None:
            positions = initial_positions.clone()
        else:
            # 在体积内随机初始化
            positions = torch.rand(N, 3, device=self.device) * 2 - 1  # [-1, 1]
            
        self.positions = nn.Parameter(positions)
        
        # 尺度参数 (N, 3) - log空间，初始化为较大尺度覆盖更多区域
        log_scale = torch.ones(N, 3, device=self.device) * np.log(self.initial_scale)
        self.scales = nn.Parameter(log_scale)
        
        # 旋转参数 (N, 4) - 四元数，初始化为单位四元数
        rotations = torch.zeros(N, 4, device=self.device)
        rotations[:, 0] = 1.0  # w = 1, x = y = z = 0
        self.rotations = nn.Parameter(rotations)
        
        # 密度参数 - 复数分为实部和虚部
        if initial_densities is not None:
            density_real = initial_densities.real.clone()
            density_imag = initial_densities.imag.clone()
        else:
            density_real = torch.randn(N, device=self.device) * 0.1
            density_imag = torch.randn(N, device=self.device) * 0.1
            
        self.density_real = nn.Parameter(density_real)
        self.density_imag = nn.Parameter(density_imag)
        
    @property
    def density(self) -> torch.Tensor:
        """获取复数密度"""
        return torch.complex(self.density_real, self.density_imag)
    
    def get_covariance(self) -> torch.Tensor:
        """
        获取协方差矩阵
        
        Returns:
            协方差矩阵 (N, 3, 3)
        """
        return build_covariance_matrix(self.scales, self.rotations)
    
    def get_scale_values(self) -> torch.Tensor:
        """获取尺度值（非log空间）"""
        return torch.exp(self.scales)
    
    def get_positions_grid(self) -> torch.Tensor:
        """
        将归一化位置转换为网格坐标
        
        Returns:
            网格坐标 (N, 3), 范围 [0, D/H/W]
        """
        D, H, W = self.volume_shape
        # 从 [-1, 1] 映射到 [0, D/H/W]
        positions_grid = (self.positions + 1) / 2  # [0, 1]
        positions_grid = positions_grid * torch.tensor(
            [D, H, W], device=self.device, dtype=torch.float32
        )
        return positions_grid
    
    def get_gaussian_params(self) -> Dict[str, torch.Tensor]:
        """获取所有Gaussian参数"""
        return {
            'positions': self.positions,
            'positions_grid': self.get_positions_grid(),
            'scales': self.get_scale_values(),
            'rotations': self.rotations,
            'covariance': self.get_covariance(),
            'density': self.density,
        }
    
    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float = 0.0002,
        scale_threshold: float = 0.01,
        use_long_axis_splitting: bool = True
    ) -> int:
        """
        Densification: 分裂大的Gaussian
        
        论文Section II-D Adaptive Control:
        当梯度超过阈值时，将大的Gaussian分裂成小的
        
        Long-axis splitting (论文Section IV-B):
        对于高加速因子，使用沿最长轴分裂的策略
        
        Args:
            grads: 位置梯度的范数 (N,)
            grad_threshold: 梯度阈值
            scale_threshold: 尺度阈值
            use_long_axis_splitting: 是否使用long-axis splitting
            
        Returns:
            新增的点数
        """
        with torch.no_grad():
            scales = self.get_scale_values()  # (N, 3)
            max_scales = scales.max(dim=-1)[0]  # (N,)
            
            # 找到需要分裂的点: 梯度大 且 尺度大
            mask = (grads > grad_threshold) & (max_scales > scale_threshold)
            
            if mask.sum() == 0:
                return 0
                
            # 获取需要分裂的点的参数
            split_positions = self.positions[mask]  # (K, 3)
            split_scales = self.scales[mask]  # (K, 3)
            split_rotations = self.rotations[mask]  # (K, 4)
            split_density_real = self.density_real[mask]  # (K,)
            split_density_imag = self.density_imag[mask]  # (K,)
            
            K = split_positions.shape[0]
            
            if use_long_axis_splitting:
                # Long-axis splitting: 沿最长轴分裂
                # 找到最长轴
                scales_exp = torch.exp(split_scales)  # (K, 3)
                longest_axis = scales_exp.argmax(dim=-1)  # (K,)
                
                # 计算偏移方向（沿最长轴）
                offset = torch.zeros_like(split_positions)  # (K, 3)
                for i in range(K):
                    axis = longest_axis[i]
                    offset[i, axis] = scales_exp[i, axis] * 0.5
                
                # 创建两个新点
                new_positions = torch.cat([
                    split_positions - offset,
                    split_positions + offset
                ], dim=0)  # (2K, 3)
                
                # 新尺度减半（在log空间减去log(2)）
                new_scales = torch.cat([
                    split_scales - np.log(2),
                    split_scales - np.log(2)
                ], dim=0)
                
                new_rotations = torch.cat([split_rotations, split_rotations], dim=0)
                new_density_real = torch.cat([split_density_real/2, split_density_real/2], dim=0)
                new_density_imag = torch.cat([split_density_imag/2, split_density_imag/2], dim=0)
                
            else:
                # 原始splitting策略
                # 在原位置周围随机偏移创建两个新点
                std = torch.exp(split_scales)
                offset = torch.randn(K, 3, device=self.device) * std * 0.5
                
                new_positions = torch.cat([
                    split_positions - offset,
                    split_positions + offset
                ], dim=0)
                
                new_scales = torch.cat([
                    split_scales - np.log(1.6),
                    split_scales - np.log(1.6)
                ], dim=0)
                
                new_rotations = torch.cat([split_rotations, split_rotations], dim=0)
                new_density_real = torch.cat([split_density_real/2, split_density_real/2], dim=0)
                new_density_imag = torch.cat([split_density_imag/2, split_density_imag/2], dim=0)
            
            # 移除原来的点，添加新点
            keep_mask = ~mask
            
            self.positions.data = torch.cat([
                self.positions[keep_mask],
                new_positions
            ], dim=0)
            
            self.scales.data = torch.cat([
                self.scales[keep_mask],
                new_scales
            ], dim=0)
            
            self.rotations.data = torch.cat([
                self.rotations[keep_mask],
                new_rotations
            ], dim=0)
            
            self.density_real.data = torch.cat([
                self.density_real[keep_mask],
                new_density_real
            ], dim=0)
            
            self.density_imag.data = torch.cat([
                self.density_imag[keep_mask],
                new_density_imag
            ], dim=0)
            
            self.num_points = self.positions.shape[0]
            
            return K  # 返回分裂的点数（实际新增K个点）
    
    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float = 0.0002,
        scale_threshold: float = 0.01
    ) -> int:
        """
        Densification: 克隆小的Gaussian
        
        当梯度超过阈值但尺度较小时，复制该Gaussian
        
        论文Section IV-B: 对于高加速因子，禁用cloning
        
        Args:
            grads: 位置梯度的范数 (N,)
            grad_threshold: 梯度阈值
            scale_threshold: 尺度阈值
            
        Returns:
            新增的点数
        """
        with torch.no_grad():
            scales = self.get_scale_values()
            max_scales = scales.max(dim=-1)[0]
            
            # 找到需要克隆的点: 梯度大 但 尺度小
            mask = (grads > grad_threshold) & (max_scales <= scale_threshold)
            
            if mask.sum() == 0:
                return 0
            
            # 克隆参数
            new_positions = self.positions[mask].clone()
            new_scales = self.scales[mask].clone()
            new_rotations = self.rotations[mask].clone()
            new_density_real = self.density_real[mask].clone()
            new_density_imag = self.density_imag[mask].clone()
            
            # 添加新点
            self.positions.data = torch.cat([self.positions, new_positions], dim=0)
            self.scales.data = torch.cat([self.scales, new_scales], dim=0)
            self.rotations.data = torch.cat([self.rotations, new_rotations], dim=0)
            self.density_real.data = torch.cat([self.density_real, new_density_real], dim=0)
            self.density_imag.data = torch.cat([self.density_imag, new_density_imag], dim=0)
            
            num_cloned = mask.sum().item()
            self.num_points = self.positions.shape[0]
            
            return num_cloned
    
    def prune(self, opacity_threshold: float = 0.01):
        """
        剪枝: 移除密度值很小的Gaussian
        
        Args:
            opacity_threshold: 密度阈值
        """
        with torch.no_grad():
            density_mag = torch.abs(self.density)
            mask = density_mag > opacity_threshold
            
            if mask.sum() == self.num_points:
                return 0  # 没有需要剪除的
                
            self.positions.data = self.positions[mask]
            self.scales.data = self.scales[mask]
            self.rotations.data = self.rotations[mask]
            self.density_real.data = self.density_real[mask]
            self.density_imag.data = self.density_imag[mask]
            
            num_pruned = self.num_points - mask.sum().item()
            self.num_points = self.positions.shape[0]
            
            return num_pruned
    
    @classmethod
    def from_image(
        cls,
        image: torch.Tensor,
        num_points: int,
        initial_scale: float = 2.0,
        device: str = "cuda:0"
    ) -> "GaussianModel3D":
        """
        从图像初始化Gaussian模型
        
        论文Section II-C:
        使用iFFT重建的图像进行初始化
        按图像强度采样初始点位置
        
        Args:
            image: 复值图像 (D, H, W)
            num_points: Gaussian点数
            initial_scale: 初始尺度
            device: 计算设备
            
        Returns:
            初始化的GaussianModel3D
        """
        volume_shape = image.shape
        D, H, W = volume_shape
        
        # 根据图像强度采样点
        image_mag = torch.abs(image).flatten()
        
        # 确保概率非负
        probs = image_mag / (image_mag.sum() + 1e-8)
        
        # 采样索引
        indices = torch.multinomial(probs, num_points, replacement=True)
        
        # 转换为3D坐标
        z_idx = indices // (H * W)
        y_idx = (indices % (H * W)) // W
        x_idx = indices % W
        
        # 归一化到 [-1, 1]
        positions = torch.stack([
            z_idx.float() / D * 2 - 1,
            y_idx.float() / H * 2 - 1,
            x_idx.float() / W * 2 - 1
        ], dim=-1).to(device)
        
        # 获取对应位置的图像值作为初始密度
        densities = image.flatten()[indices].to(device)
        
        return cls(
            num_points=num_points,
            volume_shape=volume_shape,
            initial_positions=positions,
            initial_densities=densities,
            initial_scale=initial_scale,
            device=device
        )
    
    def state_dict_custom(self) -> Dict[str, torch.Tensor]:
        """获取自定义state dict用于保存"""
        return {
            'positions': self.positions.data,
            'scales': self.scales.data,
            'rotations': self.rotations.data,
            'density_real': self.density_real.data,
            'density_imag': self.density_imag.data,
            'volume_shape': torch.tensor(self.volume_shape),
            'num_points': torch.tensor(self.num_points)
        }
    
    def load_state_dict_custom(self, state_dict: Dict[str, torch.Tensor]):
        """从自定义state dict加载"""
        self.positions = nn.Parameter(state_dict['positions'])
        self.scales = nn.Parameter(state_dict['scales'])
        self.rotations = nn.Parameter(state_dict['rotations'])
        self.density_real = nn.Parameter(state_dict['density_real'])
        self.density_imag = nn.Parameter(state_dict['density_imag'])
        self.volume_shape = tuple(state_dict['volume_shape'].tolist())
        self.num_points = state_dict['num_points'].item()
    
    def get_optimizable_params(
        self,
        lr_position: float = 1e-4,
        lr_density: float = 1e-3,
        lr_scale: float = 5e-4,
        lr_rotation: float = 1e-4
    ) -> list:
        """
        获取优化器的参数列表，支持不同的学习率
        
        Args:
            lr_position: 位置参数的学习率
            lr_density: 密度参数的学习率
            lr_scale: 尺度参数的学习率
            lr_rotation: 旋转参数的学习率
            
        Returns:
            参数组列表，可直接传递给torch.optim.Optimizer
        """
        params = [
            {'params': [self.positions], 'lr': lr_position},
            {'params': [self.scales], 'lr': lr_scale},
            {'params': [self.rotations], 'lr': lr_rotation},
            {'params': [self.density_real], 'lr': lr_density},
            {'params': [self.density_imag], 'lr': lr_density},
        ]
        return params
    
    def get_scales(self) -> torch.Tensor:
        """获取实际的尺度值（从log空间转换）"""
        return torch.exp(self.scales)
    
    def get_densities(self) -> torch.Tensor:
        """获取密度值（复数）"""
        return torch.complex(self.density_real, self.density_imag)

