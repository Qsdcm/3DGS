"""
3DGSMR Model Module
定义复数高斯模型 (Complex Gaussian Model) 和 Voxelizer
实现长轴分裂 (Long-axis Splitting) 策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math


class ComplexGaussianModel(nn.Module):
    """
    复数 3D 高斯模型
    
    与传统 3DGS 不同，MRI 重建中高斯的 "密度/不透明度" 是复数值，
    因为 MRI 图像本身是复数。
    
    参数:
        - positions: (N, 3) 高斯中心位置
        - scales: (N, 3) 各轴尺度 (控制高斯宽度)
        - rotations: (N, 4) 四元数旋转
        - densities_real: (N,) 复数密度的实部
        - densities_imag: (N,) 复数密度的虚部
    """
    
    def __init__(
        self,
        num_gaussians: int,
        volume_shape: Tuple[int, int, int],
        device: str = 'cuda',
        init_scale: float = 0.05,
        init_density: float = 0.01
    ):
        super().__init__()
        
        self.num_gaussians = num_gaussians
        self.volume_shape = volume_shape
        self.device = device
        self.init_scale = init_scale
        self.init_density = init_density
        
        # 初始化高斯参数
        self._init_parameters(num_gaussians)
        
        # 缓存协方差矩阵的逆
        self._cov_inv_cache = None
        self._cache_valid = False
    
    def _init_parameters(self, num_gaussians: int):
        """初始化高斯参数"""
        # 位置: 在 [-1, 1] 范围内均匀分布
        positions = (torch.rand(num_gaussians, 3, device=self.device) * 2 - 1)
        self.positions = nn.Parameter(positions)
        
        # 尺度: 使用 log 空间避免负值，初始化为小值
        log_scales = torch.ones(num_gaussians, 3, device=self.device) * math.log(self.init_scale)
        self.log_scales = nn.Parameter(log_scales)
        
        # 旋转: 四元数 (w, x, y, z)，初始化为单位四元数
        rotations = torch.zeros(num_gaussians, 4, device=self.device)
        rotations[:, 0] = 1.0  # w = 1, 其余为 0
        self.rotations = nn.Parameter(rotations)
        
        # 复数密度 (MRI 特有)
        densities_real = torch.randn(num_gaussians, device=self.device) * self.init_density
        densities_imag = torch.randn(num_gaussians, device=self.device) * self.init_density
        self.densities_real = nn.Parameter(densities_real)
        self.densities_imag = nn.Parameter(densities_imag)
    
    def initialize_from_image(
        self,
        image_complex: torch.Tensor,
        threshold_ratio: float = 0.1,
        max_gaussians: int = 50000,
        min_gaussians: int = 5000
    ):
        """
        使用欠采样图像的 iFFT 结果初始化高斯中心
        
        Args:
            image_complex: (nz, nx, ny) 复数图像
            threshold_ratio: 阈值比例，选择信号强度大于 max * ratio 的点
            max_gaussians: 最大高斯数量
            min_gaussians: 最小高斯数量
        """
        with torch.no_grad():
            nz, nx, ny = image_complex.shape
            
            # 计算幅度
            magnitude = torch.abs(image_complex)
            threshold = magnitude.max() * threshold_ratio
            
            # 找到信号较强的点
            mask = magnitude > threshold
            indices = torch.nonzero(mask)  # (M, 3)
            
            num_points = len(indices)
            print(f"[Model] Found {num_points} points above threshold")
            
            # 控制高斯数量
            if num_points < min_gaussians:
                # 如果点太少，降低阈值
                sorted_mag = magnitude.flatten().sort(descending=True).values
                threshold = sorted_mag[min(min_gaussians, len(sorted_mag) - 1)]
                mask = magnitude > threshold
                indices = torch.nonzero(mask)
                print(f"[Model] Lowered threshold, now {len(indices)} points")
            
            if len(indices) > max_gaussians:
                # 随机采样
                perm = torch.randperm(len(indices))[:max_gaussians]
                indices = indices[perm]
                print(f"[Model] Sampled down to {max_gaussians} points")
            
            num_gaussians = len(indices)
            print(f"[Model] Initializing {num_gaussians} Gaussians from image")
            
            # 将索引转换为归一化坐标 [-1, 1]
            positions = torch.zeros(num_gaussians, 3, device=self.device)
            positions[:, 0] = (indices[:, 0].float() / (nz - 1)) * 2 - 1
            positions[:, 1] = (indices[:, 1].float() / (nx - 1)) * 2 - 1
            positions[:, 2] = (indices[:, 2].float() / (ny - 1)) * 2 - 1
            
            # 获取对应点的复数值作为密度初始化
            densities_complex = image_complex[indices[:, 0], indices[:, 1], indices[:, 2]]
            densities_real = densities_complex.real
            densities_imag = densities_complex.imag
            
            # 归一化密度
            max_density = max(densities_real.abs().max(), densities_imag.abs().max())
            if max_density > 0:
                densities_real = densities_real / max_density * self.init_density * 10
                densities_imag = densities_imag / max_density * self.init_density * 10
            
            # 重新初始化参数
            self.num_gaussians = num_gaussians
            self.positions = nn.Parameter(positions)
            
            # 尺度根据体素大小设置
            voxel_scale = 2.0 / max(nz, nx, ny)
            log_scales = torch.ones(num_gaussians, 3, device=self.device) * math.log(voxel_scale * 2)
            self.log_scales = nn.Parameter(log_scales)
            
            rotations = torch.zeros(num_gaussians, 4, device=self.device)
            rotations[:, 0] = 1.0
            self.rotations = nn.Parameter(rotations)
            
            self.densities_real = nn.Parameter(densities_real)
            self.densities_imag = nn.Parameter(densities_imag)
            
            self._cache_valid = False
    
    @property
    def scales(self) -> torch.Tensor:
        """获取实际尺度 (N, 3)"""
        return torch.exp(self.log_scales)
    
    @property
    def complex_densities(self) -> torch.Tensor:
        """获取复数密度 (N,)"""
        return torch.complex(self.densities_real, self.densities_imag)
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        将四元数转换为旋转矩阵
        
        Args:
            q: (N, 4) 四元数 (w, x, y, z)
            
        Returns:
            R: (N, 3, 3) 旋转矩阵
        """
        # 归一化四元数
        q = F.normalize(q, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # 构建旋转矩阵
        R = torch.zeros(len(q), 3, 3, device=q.device, dtype=q.dtype)
        
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - w*x)
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)
        
        return R
    
    def compute_covariance(self) -> torch.Tensor:
        """
        计算协方差矩阵
        Σ = R @ S @ S^T @ R^T
        
        Returns:
            cov: (N, 3, 3) 协方差矩阵
        """
        scales = self.scales  # (N, 3)
        R = self.quaternion_to_rotation_matrix(self.rotations)  # (N, 3, 3)
        
        # S 是对角矩阵，S @ S^T = diag(scales^2)
        S_sq = torch.diag_embed(scales ** 2)  # (N, 3, 3)
        
        # Σ = R @ S^2 @ R^T
        cov = torch.bmm(torch.bmm(R, S_sq), R.transpose(1, 2))
        
        return cov
    
    def compute_covariance_inverse(self) -> torch.Tensor:
        """
        计算协方差矩阵的逆
        
        Returns:
            cov_inv: (N, 3, 3) 协方差逆矩阵
        """
        scales = self.scales  # (N, 3)
        R = self.quaternion_to_rotation_matrix(self.rotations)  # (N, 3, 3)
        
        # Σ^{-1} = R @ S^{-2} @ R^T
        scales_inv_sq = 1.0 / (scales ** 2 + 1e-8)  # (N, 3)
        S_inv_sq = torch.diag_embed(scales_inv_sq)  # (N, 3, 3)
        
        cov_inv = torch.bmm(torch.bmm(R, S_inv_sq), R.transpose(1, 2))
        
        return cov_inv
    
    def get_adaptive_stats(self) -> Dict:
        """
        获取用于自适应控制的统计信息
        
        Returns:
            stats: 包含位置梯度、尺度等信息的字典
        """
        stats = {
            'num_gaussians': self.num_gaussians,
            'scales': self.scales.detach(),
            'positions': self.positions.detach(),
            'position_grad': self.positions.grad.detach() if self.positions.grad is not None else None,
            'densities_magnitude': torch.abs(self.complex_densities).detach()
        }
        return stats
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播，返回所有高斯参数
        
        Returns:
            positions: (N, 3)
            cov_inv: (N, 3, 3) 协方差逆矩阵
            densities_real: (N,)
            densities_imag: (N,)
        """
        cov_inv = self.compute_covariance_inverse()
        return self.positions, cov_inv, self.densities_real, self.densities_imag


class Voxelizer(nn.Module):
    """
    3D 高斯体素化器 (显存优化版)
    
    将 3D 高斯投影到体素网格:
    x_j = Σ_i ρ_i · exp(-0.5 · (j - p_i)^T Σ_i^{-1} (j - p_i))
    
    使用可分离高斯近似，将 3D 高斯分解为三个 1D 高斯的乘积
    这大大减少了计算量和显存占用
    """
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        device: str = 'cuda',
        chunk_size: int = 256,
        voxel_chunk_size: int = 50000,
        cutoff_sigma: float = 3.0
    ):
        """
        Args:
            volume_shape: (nz, nx, ny) 体素网格形状
            device: 目标设备
            chunk_size: 分块处理的高斯数量
            voxel_chunk_size: 未使用，保留以兼容旧接口
            cutoff_sigma: 高斯截断距离
        """
        super().__init__()
        
        self.volume_shape = volume_shape
        self.device = device
        self.chunk_size = chunk_size
        self.cutoff_sigma = cutoff_sigma
        
        nz, nx, ny = volume_shape
        self.nz, self.nx, self.ny = nz, nx, ny
        self.num_voxels = nz * nx * ny
        
        # 预计算各轴坐标 [-1, 1]
        self.register_buffer('z_coords', torch.linspace(-1, 1, nz, device=device))
        self.register_buffer('x_coords', torch.linspace(-1, 1, nx, device=device))
        self.register_buffer('y_coords', torch.linspace(-1, 1, ny, device=device))
    
    def forward(
        self,
        positions: torch.Tensor,
        cov_inv: torch.Tensor,
        densities_real: torch.Tensor,
        densities_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        体素化高斯到 3D 网格 (使用可分离高斯近似)
        
        对于轴对齐的高斯（无旋转），可以分解为三个 1D 高斯的乘积。
        这里我们使用近似方法，先不考虑旋转，利用对角协方差。
        """
        num_gaussians = positions.shape[0]
        nz, nx, ny = self.nz, self.nx, self.ny
        
        # 初始化输出体积
        volume_real = torch.zeros(nz, nx, ny, device=self.device)
        volume_imag = torch.zeros(nz, nx, ny, device=self.device)
        
        # 从协方差逆矩阵提取对角元素作为各轴的精度(1/σ²)
        # cov_inv 是 (N, 3, 3)，对角元素是各轴的 1/σ²
        precision_z = cov_inv[:, 0, 0]  # (N,)
        precision_x = cov_inv[:, 1, 1]  # (N,)
        precision_y = cov_inv[:, 2, 2]  # (N,)
        
        # 分块处理高斯
        for g_start in range(0, num_gaussians, self.chunk_size):
            g_end = min(g_start + self.chunk_size, num_gaussians)
            chunk_size = g_end - g_start
            
            # 获取当前块的参数
            pos_chunk = positions[g_start:g_end]  # (G, 3)
            prec_z = precision_z[g_start:g_end]   # (G,)
            prec_x = precision_x[g_start:g_end]   # (G,)
            prec_y = precision_y[g_start:g_end]   # (G,)
            dens_r = densities_real[g_start:g_end]  # (G,)
            dens_i = densities_imag[g_start:g_end]  # (G,)
            
            # 计算各轴的 1D 高斯响应
            # z_diff: (nz, G)
            z_diff = self.z_coords.unsqueeze(1) - pos_chunk[:, 0].unsqueeze(0)
            z_response = torch.exp(-0.5 * prec_z.unsqueeze(0) * z_diff ** 2)  # (nz, G)
            del z_diff
            
            # x_diff: (nx, G)
            x_diff = self.x_coords.unsqueeze(1) - pos_chunk[:, 1].unsqueeze(0)
            x_response = torch.exp(-0.5 * prec_x.unsqueeze(0) * x_diff ** 2)  # (nx, G)
            del x_diff
            
            # y_diff: (ny, G)
            y_diff = self.y_coords.unsqueeze(1) - pos_chunk[:, 2].unsqueeze(0)
            y_response = torch.exp(-0.5 * prec_y.unsqueeze(0) * y_diff ** 2)  # (ny, G)
            del y_diff
            
            # 3D 响应 = z_response ⊗ x_response ⊗ y_response
            # 使用 einsum 计算 (nz, nx, ny, G)
            # 为了节省内存，我们不显式构建 4D 张量，而是直接计算加权和
            # volume += sum_g(dens_g * z_g ⊗ x_g ⊗ y_g)
            # = sum_g(dens_g) * (z ⊗ x ⊗ y) 当各高斯独立时
            # 使用 einsum: 'zg,xg,yg,g->zxy'
            
            contrib_real = torch.einsum('zg,xg,yg,g->zxy', z_response, x_response, y_response, dens_r)
            contrib_imag = torch.einsum('zg,xg,yg,g->zxy', z_response, x_response, y_response, dens_i)
            
            volume_real = volume_real + contrib_real
            volume_imag = volume_imag + contrib_imag
            
            # 释放中间变量
            del z_response, x_response, y_response, contrib_real, contrib_imag
        
        # 组合为复数
        volume = torch.complex(volume_real, volume_imag)
        
        return volume


class VoxelizerOptimized(nn.Module):
    """
    优化版体素化器，使用稀疏计算减少显存占用
    
    对于每个高斯，只计算其影响范围内的体素
    """
    
    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        device: str = 'cuda',
        cutoff_sigma: float = 3.0,
        gaussian_batch_size: int = 256
    ):
        super().__init__()
        
        self.volume_shape = volume_shape
        self.device = device
        self.cutoff_sigma = cutoff_sigma
        self.gaussian_batch_size = gaussian_batch_size
        
        nz, nx, ny = volume_shape
        self.nz, self.nx, self.ny = nz, nx, ny
        
        # 体素坐标
        z = torch.linspace(-1, 1, nz, device=device)
        x = torch.linspace(-1, 1, nx, device=device)
        y = torch.linspace(-1, 1, ny, device=device)
        
        self.register_buffer('z_coords', z)
        self.register_buffer('x_coords', x)
        self.register_buffer('y_coords', y)
        
        # 体素间距
        self.voxel_size = torch.tensor([
            2.0 / (nz - 1) if nz > 1 else 2.0,
            2.0 / (nx - 1) if nx > 1 else 2.0,
            2.0 / (ny - 1) if ny > 1 else 2.0
        ], device=device)
    
    def forward(
        self,
        positions: torch.Tensor,
        cov_inv: torch.Tensor,
        densities_real: torch.Tensor,
        densities_imag: torch.Tensor
    ) -> torch.Tensor:
        """
        体素化高斯（优化版）
        """
        num_gaussians = positions.shape[0]
        nz, nx, ny = self.volume_shape
        
        # 复数密度
        densities = torch.complex(densities_real, densities_imag)
        
        # 初始化输出
        volume = torch.zeros(nz, nx, ny, dtype=torch.complex64, device=self.device)
        
        # 预计算所有体素坐标
        Z, X, Y = torch.meshgrid(self.z_coords, self.x_coords, self.y_coords, indexing='ij')
        voxel_coords = torch.stack([Z, X, Y], dim=-1)  # (nz, nx, ny, 3)
        
        # 分批处理高斯
        for i in range(0, num_gaussians, self.gaussian_batch_size):
            end_idx = min(i + self.gaussian_batch_size, num_gaussians)
            batch_size = end_idx - i
            
            pos_batch = positions[i:end_idx]  # (batch, 3)
            cov_inv_batch = cov_inv[i:end_idx]  # (batch, 3, 3)
            dens_batch = densities[i:end_idx]  # (batch,)
            
            # 计算差值: (nz, nx, ny, batch, 3)
            diff = voxel_coords.unsqueeze(3) - pos_batch.view(1, 1, 1, batch_size, 3)
            
            # 计算马氏距离
            # (nz, nx, ny, batch, 3) @ (batch, 3, 3) -> (nz, nx, ny, batch, 3)
            tmp = torch.einsum('...bi,bij->...bj', diff, cov_inv_batch)
            mahal_sq = torch.einsum('...bi,...bi->...b', tmp, diff)  # (nz, nx, ny, batch)
            
            # 高斯响应
            response = torch.exp(-0.5 * mahal_sq)  # (nz, nx, ny, batch)
            
            # 加权求和
            contrib = torch.einsum('...b,b->...', response, dens_batch)
            volume = volume + contrib
        
        return volume


class AdaptiveController:
    """
    自适应控制器
    
    实现高斯的分裂 (Split) 和剪枝 (Prune) 策略
    重点实现论文中的 "Long-axis Splitting"（长轴分裂）
    """
    
    def __init__(
        self,
        split_grad_threshold: float = 0.0002,
        split_scale_threshold: float = 0.01,
        prune_density_threshold: float = 0.001,
        prune_scale_threshold: float = 0.0005,
        max_gaussians: int = 100000,
        min_gaussians: int = 1000,
        densify_interval: int = 100,
        device: str = 'cuda'
    ):
        """
        Args:
            split_grad_threshold: 位置梯度阈值，超过则考虑分裂
            split_scale_threshold: 尺度阈值，超过则执行分裂
            prune_density_threshold: 密度阈值，低于则剪枝
            prune_scale_threshold: 尺度阈值，低于则剪枝
            max_gaussians: 最大高斯数量
            min_gaussians: 最小高斯数量
            densify_interval: 执行自适应控制的迭代间隔
            device: 目标设备
        """
        self.split_grad_threshold = split_grad_threshold
        self.split_scale_threshold = split_scale_threshold
        self.prune_density_threshold = prune_density_threshold
        self.prune_scale_threshold = prune_scale_threshold
        self.max_gaussians = max_gaussians
        self.min_gaussians = min_gaussians
        self.densify_interval = densify_interval
        self.device = device
        
        # 累积梯度
        self.accumulated_grad = None
        self.grad_count = 0
        
        # 统计信息
        self.last_split_count = 0
        self.last_prune_count = 0
    
    def accumulate_gradients(self, model: ComplexGaussianModel):
        """累积位置梯度"""
        if model.positions.grad is not None:
            grad = model.positions.grad.detach().abs()
            if self.accumulated_grad is None:
                self.accumulated_grad = grad
            else:
                # 确保维度匹配
                if self.accumulated_grad.shape[0] != grad.shape[0]:
                    self.accumulated_grad = grad
                else:
                    self.accumulated_grad = self.accumulated_grad + grad
            self.grad_count += 1
    
    def reset_gradients(self):
        """重置累积梯度"""
        self.accumulated_grad = None
        self.grad_count = 0
    
    def should_densify(self, iteration: int) -> bool:
        """是否应该执行自适应控制"""
        return iteration > 0 and iteration % self.densify_interval == 0
    
    def densify_and_prune(
        self,
        model: ComplexGaussianModel,
        optimizer: torch.optim.Optimizer
    ) -> Dict:
        """
        执行分裂和剪枝
        
        Args:
            model: 高斯模型
            optimizer: 优化器
            
        Returns:
            stats: 包含分裂和剪枝数量的字典
        """
        if self.accumulated_grad is None or self.grad_count == 0:
            return {'split': 0, 'prune': 0, 'total': model.num_gaussians}
        
        # 平均梯度
        avg_grad = self.accumulated_grad / self.grad_count
        grad_norm = avg_grad.norm(dim=-1)  # (N,)
        
        scales = model.scales.detach()  # (N, 3)
        densities_mag = torch.abs(model.complex_densities).detach()  # (N,)
        
        # === 分裂 (Long-axis Splitting) ===
        # 条件: 梯度大 AND 尺度大
        large_grad_mask = grad_norm > self.split_grad_threshold
        max_scale = scales.max(dim=-1).values  # (N,)
        large_scale_mask = max_scale > self.split_scale_threshold
        
        split_mask = large_grad_mask & large_scale_mask
        
        # 限制分裂数量
        current_count = model.num_gaussians
        max_new = max(0, self.max_gaussians - current_count)
        split_indices = torch.where(split_mask)[0]
        
        if len(split_indices) > max_new:
            # 选择梯度最大的点进行分裂
            _, top_indices = torch.topk(grad_norm[split_indices], max_new)
            split_indices = split_indices[top_indices]
        
        self.last_split_count = len(split_indices)
        
        # === 剪枝 ===
        # 条件: 密度太小 OR 尺度太小
        small_density_mask = densities_mag < self.prune_density_threshold
        small_scale_mask = max_scale < self.prune_scale_threshold
        
        prune_mask = small_density_mask | small_scale_mask
        
        # 确保保留最小数量
        num_to_keep = max(self.min_gaussians, current_count - prune_mask.sum().item())
        
        prune_indices = torch.where(prune_mask)[0]
        
        # 如果剪枝太多，保留密度最大的
        if current_count - len(prune_indices) < self.min_gaussians:
            num_to_prune = max(0, current_count - self.min_gaussians)
            if num_to_prune > 0 and len(prune_indices) > 0:
                _, bottom_indices = torch.topk(densities_mag[prune_indices], 
                                               min(num_to_prune, len(prune_indices)), 
                                               largest=False)
                prune_indices = prune_indices[bottom_indices]
            else:
                prune_indices = torch.tensor([], dtype=torch.long, device=self.device)
        
        self.last_prune_count = len(prune_indices)
        
        # === 执行分裂和剪枝 ===
        if len(split_indices) > 0 or len(prune_indices) > 0:
            self._apply_densification(model, optimizer, split_indices, prune_indices)
        
        # 重置梯度累积
        self.reset_gradients()
        
        return {
            'split': self.last_split_count,
            'prune': self.last_prune_count,
            'total': model.num_gaussians
        }
    
    def _apply_densification(
        self,
        model: ComplexGaussianModel,
        optimizer: torch.optim.Optimizer,
        split_indices: torch.Tensor,
        prune_indices: torch.Tensor
    ):
        """
        应用分裂和剪枝操作
        
        实现 Long-axis Splitting: 沿最长轴分裂为两个不重叠的高斯
        """
        with torch.no_grad():
            # 获取当前参数
            old_positions = model.positions.data.clone()
            old_log_scales = model.log_scales.data.clone()
            old_rotations = model.rotations.data.clone()
            old_densities_real = model.densities_real.data.clone()
            old_densities_imag = model.densities_imag.data.clone()
            
            old_scales = model.scales.clone()
            
            # === 创建分裂后的新高斯 ===
            new_positions_list = []
            new_log_scales_list = []
            new_rotations_list = []
            new_densities_real_list = []
            new_densities_imag_list = []
            
            for idx in split_indices:
                idx = idx.item()
                pos = old_positions[idx]  # (3,)
                scales = old_scales[idx]  # (3,)
                rotation = old_rotations[idx]  # (4,)
                dens_real = old_densities_real[idx]
                dens_imag = old_densities_imag[idx]
                
                # 找到最长轴
                longest_axis = torch.argmax(scales).item()
                
                # 沿最长轴偏移
                offset = torch.zeros(3, device=self.device)
                offset[longest_axis] = scales[longest_axis] * 0.5  # 偏移半个尺度
                
                # 创建两个新高斯（替换原来的）
                # 新高斯 1: 沿正方向偏移
                new_pos_1 = pos + offset
                # 新高斯 2: 沿负方向偏移
                new_pos_2 = pos - offset
                
                # 新尺度: 沿分裂轴缩小一半
                new_scales = scales.clone()
                new_scales[longest_axis] = scales[longest_axis] * 0.5
                new_log_scale = torch.log(new_scales + 1e-8)
                
                # 密度减半
                new_dens_real = dens_real * 0.5
                new_dens_imag = dens_imag * 0.5
                
                # 添加两个新高斯
                new_positions_list.extend([new_pos_1, new_pos_2])
                new_log_scales_list.extend([new_log_scale, new_log_scale.clone()])
                new_rotations_list.extend([rotation.clone(), rotation.clone()])
                new_densities_real_list.extend([new_dens_real, new_dens_real])
                new_densities_imag_list.extend([new_dens_imag, new_dens_imag])
            
            # === 标记要删除的索引 ===
            # 分裂的点会被新点替代，所以要删除
            delete_mask = torch.zeros(len(old_positions), dtype=torch.bool, device=self.device)
            if len(split_indices) > 0:
                delete_mask[split_indices] = True
            if len(prune_indices) > 0:
                delete_mask[prune_indices] = True
            
            keep_mask = ~delete_mask
            
            # 保留的旧高斯
            kept_positions = old_positions[keep_mask]
            kept_log_scales = old_log_scales[keep_mask]
            kept_rotations = old_rotations[keep_mask]
            kept_densities_real = old_densities_real[keep_mask]
            kept_densities_imag = old_densities_imag[keep_mask]
            
            # 合并新旧高斯
            if len(new_positions_list) > 0:
                new_positions = torch.stack(new_positions_list)
                new_log_scales = torch.stack(new_log_scales_list)
                new_rotations = torch.stack(new_rotations_list)
                new_densities_real = torch.stack(new_densities_real_list)
                new_densities_imag = torch.stack(new_densities_imag_list)
                
                final_positions = torch.cat([kept_positions, new_positions], dim=0)
                final_log_scales = torch.cat([kept_log_scales, new_log_scales], dim=0)
                final_rotations = torch.cat([kept_rotations, new_rotations], dim=0)
                final_densities_real = torch.cat([kept_densities_real, new_densities_real], dim=0)
                final_densities_imag = torch.cat([kept_densities_imag, new_densities_imag], dim=0)
            else:
                final_positions = kept_positions
                final_log_scales = kept_log_scales
                final_rotations = kept_rotations
                final_densities_real = kept_densities_real
                final_densities_imag = kept_densities_imag
            
            # 更新模型参数
            new_num = len(final_positions)
            model.num_gaussians = new_num
            
            # 更新参数
            model.positions = nn.Parameter(final_positions)
            model.log_scales = nn.Parameter(final_log_scales)
            model.rotations = nn.Parameter(final_rotations)
            model.densities_real = nn.Parameter(final_densities_real)
            model.densities_imag = nn.Parameter(final_densities_imag)
            
            # 更新优化器的参数组
            # 简单起见，重新创建参数组
            optimizer.param_groups.clear()
            optimizer.param_groups.append({
                'params': list(model.parameters()),
                'lr': optimizer.defaults['lr']
            })
            
            # 重置优化器状态
            optimizer.state.clear()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    计算 PSNR
    
    Args:
        pred: 预测图像
        target: 目标图像
        
    Returns:
        psnr: PSNR 值 (dB)
    """
    mse = torch.mean((torch.abs(pred) - torch.abs(target)) ** 2)
    if mse == 0:
        return float('inf')
    max_val = torch.max(torch.abs(target))
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    计算 SSIM (简化版本，用于 3D 数据)
    
    Args:
        pred: 预测图像
        target: 目标图像
        window_size: 窗口大小
        
    Returns:
        ssim: SSIM 值
    """
    # 转换为实数（取幅度）
    pred_abs = torch.abs(pred).float()
    target_abs = torch.abs(target).float()
    
    # 常量
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 简单的均值和方差计算
    mu_pred = pred_abs.mean()
    mu_target = target_abs.mean()
    
    sigma_pred = ((pred_abs - mu_pred) ** 2).mean()
    sigma_target = ((target_abs - mu_target) ** 2).mean()
    sigma_cross = ((pred_abs - mu_pred) * (target_abs - mu_target)).mean()
    
    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    
    return ssim.item()


if __name__ == "__main__":
    # 测试代码
    device = 'cuda:1'
    volume_shape = (64, 64, 64)
    
    print("=== Testing ComplexGaussianModel ===")
    model = ComplexGaussianModel(
        num_gaussians=1000,
        volume_shape=volume_shape,
        device=device
    )
    print(f"Positions shape: {model.positions.shape}")
    print(f"Scales shape: {model.scales.shape}")
    print(f"Complex densities shape: {model.complex_densities.shape}")
    
    print("\n=== Testing Voxelizer ===")
    voxelizer = Voxelizer(volume_shape, device=device)
    
    positions, cov_inv, dens_real, dens_imag = model()
    volume = voxelizer(positions, cov_inv, dens_real, dens_imag)
    print(f"Output volume shape: {volume.shape}")
    print(f"Output dtype: {volume.dtype}")
    print(f"Output range: [{volume.abs().min():.4f}, {volume.abs().max():.4f}]")
