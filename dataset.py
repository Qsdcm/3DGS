"""
3DGSMR Dataset Module
处理 H5 读取、欠采样掩码生成（Gaussian Mask）、数据预处理
"""

import os
import numpy as np
import h5py
import torch
from typing import Tuple, Optional


def load_kspace_data(h5_path: str, device: str = 'cuda') -> torch.Tensor:
    """
    加载全采样 k-space 数据
    
    Args:
        h5_path: H5 文件路径
        device: 目标设备
        
    Returns:
        kspace: shape (nc, nz, nx, ny) 的复数张量
    """
    print(f"[Dataset] Loading k-space from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # 打印文件结构信息
        print(f"[Dataset] H5 Keys: {list(f.keys())}")
        
        # 读取 kspace 数据
        kspace = f['kspace'][:]
        print(f"[Dataset] K-space shape: {kspace.shape}, dtype: {kspace.dtype}")
    
    # 转换为 PyTorch 张量
    kspace = torch.from_numpy(kspace).to(torch.complex64).to(device)
    
    return kspace


def generate_gaussian_mask_3d(
    shape: Tuple[int, int, int],
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    生成 3D 高斯欠采样掩码
    
    掩码在 k-space 中心有更高的采样概率，符合高斯分布
    
    Args:
        shape: (nz, nx, ny) k-space 形状
        acceleration: 加速倍数
        center_fraction: 中心全采样区域的比例
        seed: 随机种子
        device: 目标设备
        
    Returns:
        mask: 二值掩码张量 shape (nz, nx, ny)
    """
    if seed is not None:
        np.random.seed(seed)
    
    nz, nx, ny = shape
    
    # 计算需要采样的点数
    total_points = nz * nx * ny
    target_samples = int(total_points / acceleration)
    
    # 中心区域全采样
    center_z = int(nz * center_fraction)
    center_x = int(nx * center_fraction)
    center_y = int(ny * center_fraction)
    
    # 确保中心尺寸至少为 1
    center_z = max(center_z, 1)
    center_x = max(center_x, 1)
    center_y = max(center_y, 1)
    
    mask = np.zeros((nz, nx, ny), dtype=np.float32)
    
    # 中心区域全采样
    z_start, z_end = nz // 2 - center_z // 2, nz // 2 + center_z // 2 + center_z % 2
    x_start, x_end = nx // 2 - center_x // 2, nx // 2 + center_x // 2 + center_x % 2
    y_start, y_end = ny // 2 - center_y // 2, ny // 2 + center_y // 2 + center_y % 2
    
    mask[z_start:z_end, x_start:x_end, y_start:y_end] = 1
    center_samples = np.sum(mask)
    
    # 剩余需要采样的点数
    remaining_samples = target_samples - int(center_samples)
    
    if remaining_samples > 0:
        # 生成 3D 高斯采样概率
        z_coords = np.linspace(-1, 1, nz)
        x_coords = np.linspace(-1, 1, nx)
        y_coords = np.linspace(-1, 1, ny)
        
        Z, X, Y = np.meshgrid(z_coords, x_coords, y_coords, indexing='ij')
        
        # 高斯分布，中心概率高
        sigma = 0.5  # 标准差控制分布宽度
        prob = np.exp(-(Z**2 + X**2 + Y**2) / (2 * sigma**2))
        
        # 排除已采样的中心区域
        prob[mask == 1] = 0
        
        # 归一化
        prob = prob / np.sum(prob)
        
        # 扁平化并采样
        prob_flat = prob.flatten()
        indices = np.arange(total_points)
        
        # 按概率采样
        sampled_indices = np.random.choice(
            indices,
            size=min(remaining_samples, int(np.sum(prob_flat > 0))),
            replace=False,
            p=prob_flat
        )
        
        # 转换回 3D 坐标
        mask_flat = mask.flatten()
        mask_flat[sampled_indices] = 1
        mask = mask_flat.reshape(nz, nx, ny)
    
    actual_rate = np.sum(mask) / total_points
    print(f"[Dataset] Gaussian mask generated: target acc={acceleration:.1f}x, "
          f"actual rate={actual_rate:.3f} ({1/actual_rate:.2f}x)")
    
    return torch.from_numpy(mask).to(device)


def generate_uniform_mask_3d(
    shape: Tuple[int, int, int],
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    生成 3D 均匀随机欠采样掩码
    
    Args:
        shape: (nz, nx, ny) k-space 形状
        acceleration: 加速倍数
        center_fraction: 中心全采样区域的比例
        seed: 随机种子
        device: 目标设备
        
    Returns:
        mask: 二值掩码张量 shape (nz, nx, ny)
    """
    if seed is not None:
        np.random.seed(seed)
    
    nz, nx, ny = shape
    
    # 计算采样率
    sampling_rate = 1.0 / acceleration
    
    # 中心区域
    center_z = int(nz * center_fraction)
    center_x = int(nx * center_fraction)
    center_y = int(ny * center_fraction)
    
    center_z = max(center_z, 1)
    center_x = max(center_x, 1)
    center_y = max(center_y, 1)
    
    # 随机生成掩码
    mask = (np.random.rand(nz, nx, ny) < sampling_rate).astype(np.float32)
    
    # 中心区域强制全采样
    z_start, z_end = nz // 2 - center_z // 2, nz // 2 + center_z // 2 + center_z % 2
    x_start, x_end = nx // 2 - center_x // 2, nx // 2 + center_x // 2 + center_x % 2
    y_start, y_end = ny // 2 - center_y // 2, ny // 2 + center_y // 2 + center_y % 2
    
    mask[z_start:z_end, x_start:x_end, y_start:y_end] = 1
    
    actual_rate = np.sum(mask) / (nz * nx * ny)
    print(f"[Dataset] Uniform mask generated: target acc={acceleration:.1f}x, "
          f"actual rate={actual_rate:.3f} ({1/actual_rate:.2f}x)")
    
    return torch.from_numpy(mask).to(device)


def apply_undersampling(
    kspace_full: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    对 k-space 应用欠采样
    
    Args:
        kspace_full: 全采样 k-space (nc, nz, nx, ny)
        mask: 采样掩码 (nz, nx, ny)
        
    Returns:
        kspace_under: 欠采样 k-space (nc, nz, nx, ny)
    """
    # 扩展 mask 维度以匹配 kspace
    mask_expanded = mask.unsqueeze(0)  # (1, nz, nx, ny)
    kspace_under = kspace_full * mask_expanded
    
    return kspace_under


def kspace_to_image(kspace: torch.Tensor) -> torch.Tensor:
    """
    将 k-space 转换为图像域 (3D iFFT)
    使用中心化正交 FFT
    
    Args:
        kspace: k-space 数据，最后三个维度是空间维度
        
    Returns:
        image: 图像域数据
    """
    # 对最后三个维度进行 3D iFFT
    axes = (-3, -2, -1)
    image = torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(kspace, dim=axes),
            dim=axes,
            norm='ortho'
        ),
        dim=axes
    )
    return image


def image_to_kspace(image: torch.Tensor) -> torch.Tensor:
    """
    将图像域转换为 k-space (3D FFT)
    使用中心化正交 FFT
    
    Args:
        image: 图像域数据，最后三个维度是空间维度
        
    Returns:
        kspace: k-space 数据
    """
    axes = (-3, -2, -1)
    kspace = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(image, dim=axes),
            dim=axes,
            norm='ortho'
        ),
        dim=axes
    )
    return kspace


def coil_combine_sos(image_multicoil: torch.Tensor) -> torch.Tensor:
    """
    使用 Sum-of-Squares (SoS) 合并多线圈图像
    
    Args:
        image_multicoil: (nc, nz, nx, ny) 多线圈复数图像
        
    Returns:
        image_combined: (nz, nx, ny) 合并后的图像（实数）
    """
    return torch.sqrt(torch.sum(torch.abs(image_multicoil) ** 2, dim=0))


def normalize_kspace(kspace: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    归一化 k-space 数据
    
    Args:
        kspace: k-space 数据
        
    Returns:
        kspace_norm: 归一化后的 k-space
        scale: 归一化系数
    """
    scale = torch.max(torch.abs(kspace)).item()
    kspace_norm = kspace / scale
    return kspace_norm, scale


class MRIDataset:
    """
    MRI 数据集类，封装数据加载和预处理
    """
    
    def __init__(
        self,
        h5_path: str,
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        mask_type: str = 'gaussian',
        seed: int = 42,
        device: str = 'cuda'
    ):
        """
        初始化数据集
        
        Args:
            h5_path: H5 数据文件路径
            acceleration: 欠采样加速倍数
            center_fraction: 中心全采样区域比例
            mask_type: 掩码类型 ('gaussian' 或 'uniform')
            seed: 随机种子
            device: 目标设备
        """
        self.device = device
        self.acceleration = acceleration
        
        # 加载全采样 k-space
        self.kspace_full = load_kspace_data(h5_path, device)
        self.nc, self.nz, self.nx, self.ny = self.kspace_full.shape
        
        print(f"[Dataset] Data shape: {self.nc} coils, volume: {self.nz}x{self.nx}x{self.ny}")
        
        # 归一化
        self.kspace_full, self.scale = normalize_kspace(self.kspace_full)
        print(f"[Dataset] Normalization scale: {self.scale:.6e}")
        
        # 生成掩码
        if mask_type == 'gaussian':
            self.mask = generate_gaussian_mask_3d(
                (self.nz, self.nx, self.ny),
                acceleration=acceleration,
                center_fraction=center_fraction,
                seed=seed,
                device=device
            )
        else:
            self.mask = generate_uniform_mask_3d(
                (self.nz, self.nx, self.ny),
                acceleration=acceleration,
                center_fraction=center_fraction,
                seed=seed,
                device=device
            )
        
        # 欠采样 k-space
        self.kspace_under = apply_undersampling(self.kspace_full, self.mask)
        
        # 计算 Ground Truth 图像 (SoS)
        image_full_multicoil = kspace_to_image(self.kspace_full)
        self.image_gt = coil_combine_sos(image_full_multicoil)
        
        # 计算零填充重建图像（用于初始化）
        image_under_multicoil = kspace_to_image(self.kspace_under)
        self.image_init = coil_combine_sos(image_under_multicoil)
        
        # 复数图像用于初始化高斯位置（取第一个线圈或平均）
        self.image_init_complex = torch.mean(image_under_multicoil, dim=0)
        
        print(f"[Dataset] GT image range: [{self.image_gt.min():.4f}, {self.image_gt.max():.4f}]")
        print(f"[Dataset] Init image range: [{self.image_init.min():.4f}, {self.image_init.max():.4f}]")
    
    def get_volume_shape(self) -> Tuple[int, int, int]:
        """获取体素网格形状"""
        return (self.nz, self.nx, self.ny)
    
    def get_normalized_coords(self) -> torch.Tensor:
        """
        获取归一化坐标网格 [-1, 1]
        
        Returns:
            coords: (nz*nx*ny, 3) 的坐标张量
        """
        z = torch.linspace(-1, 1, self.nz, device=self.device)
        x = torch.linspace(-1, 1, self.nx, device=self.device)
        y = torch.linspace(-1, 1, self.ny, device=self.device)
        
        Z, X, Y = torch.meshgrid(z, x, y, indexing='ij')
        coords = torch.stack([Z, X, Y], dim=-1).reshape(-1, 3)
        
        return coords


if __name__ == "__main__":
    # 测试代码
    dataset = MRIDataset(
        h5_path="/data/data54/wanghaobo/data/ksp_full.h5",
        acceleration=4.0,
        mask_type='gaussian',
        device='cuda:1'
    )
    
    print(f"\n=== Dataset Test ===")
    print(f"K-space full shape: {dataset.kspace_full.shape}")
    print(f"K-space under shape: {dataset.kspace_under.shape}")
    print(f"Mask shape: {dataset.mask.shape}")
    print(f"GT image shape: {dataset.image_gt.shape}")
    print(f"Init image shape: {dataset.image_init.shape}")
