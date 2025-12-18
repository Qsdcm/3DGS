"""dataset.py

数据集封装模块：扫描 data_root 下的 H5 文件，提供样本迭代接口。

支持两种模式：
1. data_root 是单个 H5 文件 -> 返回该文件
2. data_root 是目录 -> 扫描所有 *.h5 文件

用法：
    from dataset import MRIDatasetLoader
    loader = MRIDatasetLoader(data_root, acceleration=4.0, mask_type="gaussian")
    for sample_name, mri_ds in loader:
        # mri_ds 是 MRIDataset 实例
        kspace_under_single = mri_ds.kspace_under_single
        ...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from data_read import MRIDataset


class MRIDatasetLoader:
    """MRI 数据集加载器，支持单文件或目录扫描。"""

    def __init__(
        self,
        data_root: str,
        acceleration: float = 4.0,
        center_fraction: float = 0.08,
        mask_type: str = "gaussian",
        seed: int = 42,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            data_root: H5 文件路径或包含 H5 文件的目录
            acceleration: 欠采样加速倍数
            center_fraction: 中心全采样比例
            mask_type: "gaussian" 或 "uniform"
            seed: 随机种子
            device: 目标设备
            rank: 当前进程 rank（用于分布式）
            world_size: 总进程数（用于分布式）
        """
        self.data_root = data_root
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.mask_type = mask_type
        self.seed = seed
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # 获取 H5 文件列表
        self.h5_files = self._scan_h5_files()
        print(f"[DatasetLoader] Found {len(self.h5_files)} H5 files")

        # 分布式切分：按文件名排序后 stride 切分
        if world_size > 1:
            self.h5_files = self.h5_files[rank::world_size]
            print(f"[DatasetLoader] Rank {rank}/{world_size}: assigned {len(self.h5_files)} files")

    def _scan_h5_files(self) -> List[str]:
        """扫描 H5 文件列表。"""
        data_path = Path(self.data_root)

        if data_path.is_file():
            # 单文件模式
            if data_path.suffix.lower() in [".h5", ".hdf5"]:
                return [str(data_path)]
            else:
                raise ValueError(f"Not an H5 file: {data_path}")

        elif data_path.is_dir():
            # 目录模式：扫描所有 H5 文件
            h5_files = []
            for ext in ["*.h5", "*.hdf5", "*.H5", "*.HDF5"]:
                h5_files.extend(data_path.glob(ext))
            # 按文件名排序，确保分布式时各 rank 一致
            h5_files = sorted([str(f) for f in h5_files])
            if len(h5_files) == 0:
                raise ValueError(f"No H5 files found in: {data_path}")
            return h5_files

        else:
            raise ValueError(f"Invalid data_root: {data_path}")

    def __len__(self) -> int:
        return len(self.h5_files)

    def __iter__(self) -> Iterator[Tuple[str, MRIDataset]]:
        """迭代返回 (sample_name, MRIDataset) 对。"""
        for h5_path in self.h5_files:
            sample_name = Path(h5_path).stem
            print(f"\n[DatasetLoader] Loading sample: {sample_name}")

            mri_ds = MRIDataset(
                h5_path=h5_path,
                acceleration=self.acceleration,
                center_fraction=self.center_fraction,
                mask_type=self.mask_type,
                seed=self.seed,
                device=self.device,
            )
            yield sample_name, mri_ds

    def get_sample(self, index: int) -> Tuple[str, MRIDataset]:
        """获取指定索引的样本。"""
        if index < 0 or index >= len(self.h5_files):
            raise IndexError(f"Index {index} out of range [0, {len(self.h5_files)})")

        h5_path = self.h5_files[index]
        sample_name = Path(h5_path).stem

        mri_ds = MRIDataset(
            h5_path=h5_path,
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            mask_type=self.mask_type,
            seed=self.seed,
            device=self.device,
        )
        return sample_name, mri_ds

    def get_sample_names(self) -> List[str]:
        """获取所有样本名称列表。"""
        return [Path(f).stem for f in self.h5_files]


def get_sample_data(
    mri_ds: MRIDataset,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """从 MRIDataset 提取训练/测试所需数据。

    Returns:
        kspace_under_single: (nz, nx, ny) complex64 - 欠采样 k-space（单通道）
        mask: (nz, nx, ny) float32 - 采样 mask
        image_gt: (nz, nx, ny) float32 - GT 图像（SoS magnitude）
        image_init_complex: (nz, nx, ny) complex64 - 初始复数图像
        meta: dict - 元信息
    """
    import torch
    
    meta = {
        "shape": mri_ds.get_volume_shape(),
        "nc": mri_ds.nc,
        "scale": mri_ds.scale,
        "h5_path": mri_ds.h5_path,
    }

    return (
        mri_ds.kspace_under_single,
        mri_ds.mask,
        mri_ds.image_gt,
        mri_ds.image_init_complex,
        meta,
    )


# 为了在其他地方可以 import torch
import torch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loader")
    parser.add_argument("--data_root", required=True, help="H5 file or directory")
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--mask_type", type=str, default="gaussian")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    loader = MRIDatasetLoader(
        data_root=args.data_root,
        acceleration=args.acceleration,
        mask_type=args.mask_type,
        device=args.device,
    )

    print(f"\n=== Dataset Loader Test ===")
    print(f"Total samples: {len(loader)}")
    print(f"Sample names: {loader.get_sample_names()}")

    for sample_name, mri_ds in loader:
        kspace_under, mask, image_gt, image_init_complex, meta = get_sample_data(mri_ds)
        print(f"\nSample: {sample_name}")
        print(f"  kspace_under_single: {tuple(kspace_under.shape)} {kspace_under.dtype}")
        print(f"  mask: {tuple(mask.shape)} {mask.dtype}")
        print(f"  image_gt: {tuple(image_gt.shape)} {image_gt.dtype}")
        print(f"  image_init_complex: {tuple(image_init_complex.shape)} {image_init_complex.dtype}")
        print(f"  meta: {meta}")
        break  # 只测试第一个
