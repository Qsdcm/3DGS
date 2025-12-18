"""data_read.py

从本仓库的 dataset.py 抽取/整理的数据读取与预处理代码。

功能：
- 从 H5 文件读取全采样 k-space（key: 'kspace'）
- 生成 3D 欠采样 mask（gaussian / uniform）
- 应用欠采样得到 kspace_under
- 3D iFFT 转到图像域，并做 SoS 合并得到 GT/zero-fill

用法示例：
    from data_read import MRIDataset
    ds = MRIDataset("/path/to/ksp_full.h5", acceleration=4.0, mask_type="gaussian", device="cuda:0")

注意：
- H5 内必须存在 dataset: 'kspace'
- kspace 期望形状: (nc, nz, nx, ny)
- kspace 推荐为复数 dtype（如 complex64/complex128）
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import h5py
import torch


def load_kspace_data(h5_path: str, device: str = "cuda") -> torch.Tensor:
    """加载全采样 k-space 数据。

    Args:
        h5_path: H5 文件路径
        device: 目标设备

    Returns:
        kspace: shape (nc, nz, nx, ny) 的复数张量
    """
    print(f"[Dataset] Loading k-space from: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        print(f"[Dataset] H5 Keys: {list(f.keys())}")
        kspace = f["kspace"][:]
        print(f"[Dataset] K-space shape: {kspace.shape}, dtype: {kspace.dtype}")

    # 转换为 PyTorch 复数张量
    kspace_t = torch.from_numpy(kspace).to(torch.complex64).to(device)
    return kspace_t


def generate_gaussian_mask_3d(
    shape: Tuple[int, int, int],
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """生成 3D 高斯欠采样掩码。

    Args:
        shape: (nz, nx, ny)
        acceleration: 加速倍数
        center_fraction: 中心全采样比例
        seed: 随机种子
        device: 目标设备

    Returns:
        mask: (nz, nx, ny) 0/1 float 张量
    """
    if seed is not None:
        np.random.seed(seed)

    nz, nx, ny = shape
    total_points = nz * nx * ny
    target_samples = int(total_points / acceleration)

    center_z = max(int(nz * center_fraction), 1)
    center_x = max(int(nx * center_fraction), 1)
    center_y = max(int(ny * center_fraction), 1)

    mask = np.zeros((nz, nx, ny), dtype=np.float32)

    z_start, z_end = nz // 2 - center_z // 2, nz // 2 + center_z // 2 + center_z % 2
    x_start, x_end = nx // 2 - center_x // 2, nx // 2 + center_x // 2 + center_x % 2
    y_start, y_end = ny // 2 - center_y // 2, ny // 2 + center_y // 2 + center_y % 2

    mask[z_start:z_end, x_start:x_end, y_start:y_end] = 1
    center_samples = np.sum(mask)

    remaining_samples = target_samples - int(center_samples)

    if remaining_samples > 0:
        z_coords = np.linspace(-1, 1, nz)
        x_coords = np.linspace(-1, 1, nx)
        y_coords = np.linspace(-1, 1, ny)

        Z, X, Y = np.meshgrid(z_coords, x_coords, y_coords, indexing="ij")

        sigma = 0.5
        prob = np.exp(-(Z**2 + X**2 + Y**2) / (2 * sigma**2))

        prob[mask == 1] = 0
        prob = prob / np.sum(prob)

        prob_flat = prob.flatten()
        indices = np.arange(total_points)

        sampled_indices = np.random.choice(
            indices,
            size=min(remaining_samples, int(np.sum(prob_flat > 0))),
            replace=False,
            p=prob_flat,
        )

        mask_flat = mask.flatten()
        mask_flat[sampled_indices] = 1
        mask = mask_flat.reshape(nz, nx, ny)

    actual_rate = np.sum(mask) / total_points
    print(
        f"[Dataset] Gaussian mask generated: target acc={acceleration:.1f}x, "
        f"actual rate={actual_rate:.3f} ({1 / actual_rate:.2f}x)"
    )

    return torch.from_numpy(mask).to(device)


def generate_uniform_mask_3d(
    shape: Tuple[int, int, int],
    acceleration: float = 4.0,
    center_fraction: float = 0.08,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """生成 3D 均匀随机欠采样掩码。"""
    if seed is not None:
        np.random.seed(seed)

    nz, nx, ny = shape
    sampling_rate = 1.0 / acceleration

    center_z = max(int(nz * center_fraction), 1)
    center_x = max(int(nx * center_fraction), 1)
    center_y = max(int(ny * center_fraction), 1)

    mask = (np.random.rand(nz, nx, ny) < sampling_rate).astype(np.float32)

    z_start, z_end = nz // 2 - center_z // 2, nz // 2 + center_z // 2 + center_z % 2
    x_start, x_end = nx // 2 - center_x // 2, nx // 2 + center_x // 2 + center_x % 2
    y_start, y_end = ny // 2 - center_y // 2, ny // 2 + center_y // 2 + center_y % 2

    mask[z_start:z_end, x_start:x_end, y_start:y_end] = 1

    actual_rate = np.sum(mask) / (nz * nx * ny)
    print(
        f"[Dataset] Uniform mask generated: target acc={acceleration:.1f}x, "
        f"actual rate={actual_rate:.3f} ({1 / actual_rate:.2f}x)"
    )

    return torch.from_numpy(mask).to(device)


def apply_undersampling(kspace_full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 k-space 应用欠采样。

    Args:
        kspace_full: (nc, nz, nx, ny)
        mask: (nz, nx, ny)

    Returns:
        kspace_under: (nc, nz, nx, ny)
    """
    mask_expanded = mask.unsqueeze(0)  # (1, nz, nx, ny)
    return kspace_full * mask_expanded


def kspace_to_image(kspace: torch.Tensor) -> torch.Tensor:
    """将 k-space 转换为图像域 (3D iFFT)，中心化正交 FFT。"""
    axes = (-3, -2, -1)
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(kspace, dim=axes),
            dim=axes,
            norm="ortho",
        ),
        dim=axes,
    )


def image_to_kspace(image: torch.Tensor) -> torch.Tensor:
    """将图像域转换为 k-space (3D FFT)，中心化正交 FFT。"""
    axes = (-3, -2, -1)
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(image, dim=axes),
            dim=axes,
            norm="ortho",
        ),
        dim=axes,
    )


def coil_combine_sos(image_multicoil: torch.Tensor) -> torch.Tensor:
    """Sum-of-Squares (SoS) 合并多线圈图像。

    Args:
        image_multicoil: (nc, nz, nx, ny) 复数图像

    Returns:
        (nz, nx, ny) 实数图像
    """
    return torch.sqrt(torch.sum(torch.abs(image_multicoil) ** 2, dim=0))


def normalize_kspace(kspace: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """归一化 k-space（按最大幅值）。"""
    scale = torch.max(torch.abs(kspace)).item()
    return kspace / scale, scale


class MRIDataset:
    """MRI 数据集封装（单个 3D volume，一次性读入显存/内存）。"""

    def __init__(
        self,
        h5_path: str,
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        mask_type: str = "gaussian",
        seed: int = 42,
        device: str = "cuda",
    ):
        self.device = device
        self.acceleration = acceleration
        self.h5_path = h5_path

        # 1) 加载全采样 k-space
        self.kspace_full = load_kspace_data(h5_path, device)
        self.nc, self.nz, self.nx, self.ny = self.kspace_full.shape
        print(f"[Dataset] Data shape: {self.nc} coils, volume: {self.nz}x{self.nx}x{self.ny}")

        # 2) 归一化
        self.kspace_full, self.scale = normalize_kspace(self.kspace_full)
        print(f"[Dataset] Normalization scale: {self.scale:.6e}")

        # 3) mask
        if mask_type == "gaussian":
            self.mask = generate_gaussian_mask_3d(
                (self.nz, self.nx, self.ny),
                acceleration=acceleration,
                center_fraction=center_fraction,
                seed=seed,
                device=device,
            )
        else:
            self.mask = generate_uniform_mask_3d(
                (self.nz, self.nx, self.ny),
                acceleration=acceleration,
                center_fraction=center_fraction,
                seed=seed,
                device=device,
            )

        # 4) 欠采样
        self.kspace_under = apply_undersampling(self.kspace_full, self.mask)

        # 5) 单通道近似（多线圈简单均值；严格多线圈应使用 CSM）
        self.kspace_full_single = torch.mean(self.kspace_full, dim=0)  # (nz, nx, ny)
        self.kspace_under_single = torch.mean(self.kspace_under, dim=0)  # (nz, nx, ny)

        # 6) 图像域 GT / zero-fill
        image_full_multicoil = kspace_to_image(self.kspace_full)
        self.image_gt = coil_combine_sos(image_full_multicoil)

        image_under_multicoil = kspace_to_image(self.kspace_under)
        self.image_init = coil_combine_sos(image_under_multicoil)

        # 复数图像用于初始化（均值线圈）
        self.image_init_complex = torch.mean(image_under_multicoil, dim=0)

        print(f"[Dataset] GT image range: [{self.image_gt.min():.4f}, {self.image_gt.max():.4f}]")
        print(f"[Dataset] Init image range: [{self.image_init.min():.4f}, {self.image_init.max():.4f}]")

    def get_volume_shape(self) -> Tuple[int, int, int]:
        return (self.nz, self.nx, self.ny)

    def get_normalized_coords(self) -> torch.Tensor:
        """获取归一化坐标网格 [-1, 1]，shape: (nz*nx*ny, 3)。"""
        z = torch.linspace(-1, 1, self.nz, device=self.device)
        x = torch.linspace(-1, 1, self.nx, device=self.device)
        y = torch.linspace(-1, 1, self.ny, device=self.device)

        Z, X, Y = torch.meshgrid(z, x, y, indexing="ij")
        return torch.stack([Z, X, Y], dim=-1).reshape(-1, 3)


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Standalone data reader for 3DGSMR")
    p.add_argument("--h5", required=True, help="Path to H5 file containing dataset 'kspace'")
    p.add_argument("--acceleration", type=float, default=4.0)
    p.add_argument("--center_fraction", type=float, default=0.08)
    p.add_argument("--mask_type", type=str, default="gaussian", choices=["gaussian", "uniform"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ds = MRIDataset(
        h5_path=args.h5,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        mask_type=args.mask_type,
        seed=args.seed,
        device=args.device,
    )

    print("\n=== Data Read OK ===")
    print(f"kspace_full: {tuple(ds.kspace_full.shape)} {ds.kspace_full.dtype}")
    print(f"kspace_under: {tuple(ds.kspace_under.shape)} {ds.kspace_under.dtype}")
    print(f"mask: {tuple(ds.mask.shape)} {ds.mask.dtype}")
    print(f"image_gt: {tuple(ds.image_gt.shape)} {ds.image_gt.dtype}")
    print(f"image_init: {tuple(ds.image_init.shape)} {ds.image_init.dtype}")
    print(f"image_init_complex: {tuple(ds.image_init_complex.shape)} {ds.image_init_complex.dtype}")
