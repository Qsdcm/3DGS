"""
MRI Dataset for 3DGSMR
从h5文件加载多线圈k-space数据
"""

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
from .transforms import ifft3c, fft3c, normalize_kspace


class MRIDataset(Dataset):
    """
    MRI数据集类
    
    从h5文件加载k-space数据，支持：
    - 多线圈数据合并
    - k-space归一化
    - 欠采样mask生成
    """
    
    def __init__(
        self,
        data_path: str,
        acceleration_factor: int = 4,
        mask_type: str = "gaussian",
        use_acs: bool = True,
        acs_lines: int = 24,
        device: str = "cuda:0"
    ):
        """
        Args:
            data_path: h5文件路径
            acceleration_factor: 加速因子
            mask_type: 欠采样mask类型 ("gaussian", "poisson", "random")
            use_acs: 是否使用ACS区域
            acs_lines: ACS区域线数
            device: 计算设备
        """
        super().__init__()
        self.data_path = data_path
        self.acceleration_factor = acceleration_factor
        self.mask_type = mask_type
        self.use_acs = use_acs
        self.acs_lines = acs_lines
        self.device = device
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载h5文件中的k-space数据"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # 假设数据格式为 (num_coils, kx, ky, kz) - 复数
            if 'kspace' in f.keys():
                kspace_data = f['kspace'][:]
            elif 'ksp' in f.keys():
                kspace_data = f['ksp'][:]
            else:
                # 尝试获取第一个key
                key = list(f.keys())[0]
                kspace_data = f[key][:]
                print(f"Using key: {key}")
        
        # 转换为torch tensor
        # kspace_data shape: (num_coils, kx, ky, kz)
        self.kspace_multicoil = torch.from_numpy(kspace_data).to(torch.complex64)
        self.num_coils = self.kspace_multicoil.shape[0]
        self.volume_shape = self.kspace_multicoil.shape[1:]  # (kx, ky, kz)
        
        print(f"Loaded k-space data with shape: {self.kspace_multicoil.shape}")
        print(f"Number of coils: {self.num_coils}")
        print(f"Volume shape: {self.volume_shape}")
        
        # 估计线圈敏感度图 (使用简化的方法)
        self._estimate_sensitivity_maps()
        
        # 合并多线圈数据得到单通道ground truth
        self._combine_coils()
        
        # 生成欠采样mask
        self._generate_mask()
        
        # 应用欠采样
        self._apply_undersampling()
        
    def _estimate_sensitivity_maps(self):
        """
        估计线圈敏感度图
        使用简化的方法：从全采样低频区域估计
        参考: ESPIRiT / SENSE
        """
        # 简化方法：使用RSS(Root Sum of Squares)归一化
        # 首先获取图像域数据
        images_multicoil = ifft3c(self.kspace_multicoil)  # (num_coils, x, y, z)
        
        # RSS合并
        rss = torch.sqrt(torch.sum(torch.abs(images_multicoil) ** 2, dim=0))
        rss = rss + 1e-8  # 避免除零
        
        # 敏感度图 = 单线圈图像 / RSS
        self.sensitivity_maps = images_multicoil / rss.unsqueeze(0)
        
        print(f"Estimated sensitivity maps with shape: {self.sensitivity_maps.shape}")
        
    def _combine_coils(self):
        """
        合并多线圈数据得到单通道ground truth image
        使用敏感度加权合并
        """
        # 多线圈图像
        images_multicoil = ifft3c(self.kspace_multicoil)  # (num_coils, x, y, z)
        
        # 敏感度加权合并: sum(conj(S) * I) 
        self.ground_truth_image = torch.sum(
            torch.conj(self.sensitivity_maps) * images_multicoil, 
            dim=0
        )  # (x, y, z)
        
        # 对应的全采样k-space
        self.kspace_full = fft3c(self.ground_truth_image)
        
        print(f"Ground truth image shape: {self.ground_truth_image.shape}")
        
    def _generate_mask(self):
        """
        生成欠采样mask
        支持多种采样模式
        
        论文中使用Gaussian采样，保留低频中心，随机采样高频
        """
        shape = self.volume_shape
        mask = torch.zeros(shape, dtype=torch.float32)
        
        if self.mask_type == "gaussian":
            mask = self._generate_gaussian_mask(shape)
        elif self.mask_type == "poisson":
            mask = self._generate_poisson_mask(shape)
        else:  # random
            mask = self._generate_random_mask(shape)
            
        # 添加ACS区域（全采样中心区域）
        if self.use_acs:
            mask = self._add_acs_region(mask)
            
        self.mask = mask
        actual_acc = mask.numel() / mask.sum().item()
        print(f"Generated {self.mask_type} mask with actual acceleration: {actual_acc:.2f}x")
        
    def _generate_gaussian_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        生成Gaussian采样mask
        
        Gaussian采样保留更多低频，符合MRI k-space能量分布
        参考论文Section III-B
        """
        kx, ky, kz = shape
        
        # 计算需要采样的点数
        total_points = kx * ky * kz
        target_points = int(total_points / self.acceleration_factor)
        
        # 创建以k-space中心为原点的坐标网格
        cx, cy, cz = kx // 2, ky // 2, kz // 2
        
        # 采样概率：高斯分布，中心概率高
        x = torch.arange(kx) - cx
        y = torch.arange(ky) - cy
        z = torch.arange(kz) - cz
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # 计算到中心的距离
        sigma = min(kx, ky, kz) / 4  # 控制高斯宽度
        prob = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
        prob = prob / prob.sum() * target_points
        prob = torch.clamp(prob, 0, 1)
        
        # 随机采样
        mask = (torch.rand(shape) < prob).float()
        
        return mask
    
    def _generate_poisson_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成Poisson disc采样mask (简化版本)"""
        # 简化实现：使用随机采样但保证采样率
        return self._generate_random_mask(shape)
    
    def _generate_random_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成随机欠采样mask"""
        prob = 1.0 / self.acceleration_factor
        mask = (torch.rand(shape) < prob).float()
        return mask
    
    def _add_acs_region(self, mask: torch.Tensor) -> torch.Tensor:
        """添加ACS（全采样中心）区域"""
        kx, ky, kz = mask.shape
        cx, cy, cz = kx // 2, ky // 2, kz // 2
        half_acs = self.acs_lines // 2
        
        # 在中心区域设置为全采样
        mask[cx-half_acs:cx+half_acs, cy-half_acs:cy+half_acs, cz-half_acs:cz+half_acs] = 1.0
        
        return mask
        
    def _apply_undersampling(self):
        """应用欠采样"""
        # 对单通道k-space应用mask
        self.kspace_undersampled = self.kspace_full * self.mask
        
        # Zero-filled重建 (作为初始化)
        self.zero_filled_image = ifft3c(self.kspace_undersampled)
        
        print(f"Applied undersampling. Non-zero ratio: {self.mask.sum()/self.mask.numel():.4f}")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
        """
        获取处理后的数据
        
        Returns:
            包含以下键的字典:
            - kspace_full: 全采样k-space
            - kspace_undersampled: 欠采样k-space
            - mask: 欠采样mask
            - ground_truth: ground truth图像
            - zero_filled: zero-filled重建图像
            - sensitivity_maps: 线圈敏感度图
        """
        return {
            'kspace_full': self.kspace_full,
            'kspace_undersampled': self.kspace_undersampled,
            'mask': self.mask,
            'ground_truth': self.ground_truth_image,
            'zero_filled': self.zero_filled_image,
            'sensitivity_maps': self.sensitivity_maps,
            'volume_shape': self.volume_shape
        }
    
    def __len__(self):
        return 1  # 单例数据集
    
    def __getitem__(self, idx):
        return self.get_data()
