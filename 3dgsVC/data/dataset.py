"""
MRI Dataset for 3DGSMR (Fixed Mask Generation)
修正了Mask生成逻辑：从3D随机点采样改为符合MRI物理的2D相位编码方向欠采样。
这会产生正确的混叠伪影(Aliasing artifacts)而非仅仅是模糊。
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
    - 欠采样mask生成 (Fixed: 2D Phase Encoding Mask)
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
        super().__init__()
        self.data_path = data_path
        self.acceleration_factor = acceleration_factor
        self.mask_type = mask_type
        self.use_acs = use_acs
        self.acs_lines = acs_lines
        self.device = device
        
        self._load_data()
        
    def _load_data(self):
        """加载h5文件中的k-space数据"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            if 'kspace' in f.keys():
                kspace_data = f['kspace'][:]
            elif 'ksp' in f.keys():
                kspace_data = f['ksp'][:]
            else:
                key = list(f.keys())[0]
                kspace_data = f[key][:]
                print(f"Using key: {key}")
        
        # kspace_data shape: (num_coils, kx, ky, kz)
        self.kspace_multicoil = torch.from_numpy(kspace_data).to(torch.complex64)
        self.num_coils = self.kspace_multicoil.shape[0]
        self.volume_shape = self.kspace_multicoil.shape[1:]  # (kx, ky, kz)
        
        print(f"Loaded k-space data with shape: {self.kspace_multicoil.shape}")
        print(f"Number of coils: {self.num_coils}")
        print(f"Volume shape: {self.volume_shape}")
        
        self._estimate_sensitivity_maps()
        self._combine_coils()
        self._generate_mask()
        self._apply_undersampling()
        
    def _estimate_sensitivity_maps(self):
        """估计线圈敏感度图"""
        images_multicoil = ifft3c(self.kspace_multicoil)
        rss = torch.sqrt(torch.sum(torch.abs(images_multicoil) ** 2, dim=0))
        rss = rss + 1e-8
        self.sensitivity_maps = images_multicoil / rss.unsqueeze(0)
        print(f"Estimated sensitivity maps with shape: {self.sensitivity_maps.shape}")
        
    def _combine_coils(self):
        """合并多线圈数据得到单通道ground truth image"""
        images_multicoil = ifft3c(self.kspace_multicoil)
        self.ground_truth_image = torch.sum(
            torch.conj(self.sensitivity_maps) * images_multicoil, 
            dim=0
        )
        self.kspace_full = fft3c(self.ground_truth_image)
        print(f"Ground truth image shape: {self.ground_truth_image.shape}")
        
    def _generate_mask(self):
        """生成欠采样mask"""
        shape = self.volume_shape
        
        if self.mask_type == "gaussian":
            mask = self._generate_gaussian_mask(shape)
        elif self.mask_type == "poisson":
            mask = self._generate_poisson_mask(shape)
        else:  # random
            mask = self._generate_random_mask(shape)
            
        if self.use_acs:
            mask = self._add_acs_region(mask)
            
        self.mask = mask
        actual_acc = mask.numel() / mask.sum().item()
        print(f"Generated {self.mask_type} mask (2D Phase Encoding) with actual acceleration: {actual_acc:.2f}x")
        
    def _generate_gaussian_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        生成Gaussian采样mask (2D Phase Encoding)
        保留kx(Readout)全采样，只在ky-kz平面欠采样
        """
        kx, ky, kz = shape
        
        # 只在 ky-kz 平面计算
        total_lines = ky * kz
        target_lines = int(total_lines / self.acceleration_factor)
        
        cy, cz = ky // 2, kz // 2
        
        y = torch.arange(ky) - cy
        z = torch.arange(kz) - cz
        
        yy, zz = torch.meshgrid(y, z, indexing='ij')
        
        # 2D Gaussian PDF
        sigma = min(ky, kz) / 4
        prob = torch.exp(-(yy**2 + zz**2) / (2 * sigma**2))
        
        # Normalize to match target acceleration
        prob = prob / prob.sum() * target_lines
        prob = torch.clamp(prob, 0, 1)
        
        # 2D Sampling
        mask_2d = (torch.rand((ky, kz)) < prob).float()
        
        # Expand to 3D (kx is fully sampled)
        mask_3d = mask_2d.unsqueeze(0).expand(kx, ky, kz)
        
        return mask_3d
    
    def _generate_poisson_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成Poisson mask (此处简化为Random，如需严格Poisson需额外算法)"""
        return self._generate_random_mask(shape)
    
    def _generate_random_mask(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """生成随机欠采样mask (2D Phase Encoding)"""
        kx, ky, kz = shape
        prob = 1.0 / self.acceleration_factor
        
        # 2D Sampling
        mask_2d = (torch.rand((ky, kz)) < prob).float()
        
        # Expand to 3D
        mask_3d = mask_2d.unsqueeze(0).expand(kx, ky, kz)
        
        return mask_3d
    
    def _add_acs_region(self, mask: torch.Tensor) -> torch.Tensor:
        """添加ACS（全采样中心）区域"""
        kx, ky, kz = mask.shape
        cy, cz = ky // 2, kz // 2
        half_acs = self.acs_lines // 2
        
        # 只在 ky-kz 平面中心添加，贯穿所有 kx
        # 注意: mask是共享内存的，直接修改会生效
        mask[:, cy-half_acs:cy+half_acs, cz-half_acs:cz+half_acs] = 1.0
        
        return mask
        
    def _apply_undersampling(self):
        """应用欠采样并生成Zero-filled图像"""
        self.kspace_undersampled = self.kspace_full * self.mask
        self.zero_filled_image = ifft3c(self.kspace_undersampled)
        
        print(f"Applied undersampling. Non-zero ratio: {self.mask.sum()/self.mask.numel():.4f}")
        
    def get_data(self) -> Dict[str, torch.Tensor]:
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
        return 1
    
    def __getitem__(self, idx):
        return self.get_data()