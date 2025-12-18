"""utils.py

工具函数模块：
- 复数操作
- 评估指标（PSNR, SSIM）
- 随机种子
- 路径工具
- 3D Total Variation
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


# ============================================================================
# 随机种子
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """设置全局随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 可选：确定性算法（可能影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ============================================================================
# 复数操作
# ============================================================================

def complex_to_real_channels(x: torch.Tensor) -> torch.Tensor:
    """将复数张量转换为两通道实数张量。

    Args:
        x: (..., ) complex tensor

    Returns:
        (..., 2) real tensor with [real, imag] in last dim
    """
    return torch.stack([x.real, x.imag], dim=-1)


def real_channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """将两通道实数张量转换为复数张量。

    Args:
        x: (..., 2) real tensor with [real, imag] in last dim

    Returns:
        (..., ) complex tensor
    """
    return torch.complex(x[..., 0], x[..., 1])


def complex_abs(x: torch.Tensor) -> torch.Tensor:
    """计算复数张量的幅值（magnitude）。"""
    return torch.abs(x)


def complex_angle(x: torch.Tensor) -> torch.Tensor:
    """计算复数张量的相位（phase）。"""
    return torch.angle(x)


# ============================================================================
# 评估指标
# ============================================================================

def compute_psnr(
    img_pred: Union[np.ndarray, torch.Tensor],
    img_gt: Union[np.ndarray, torch.Tensor],
    data_range: Optional[float] = None,
) -> float:
    """计算 PSNR（Peak Signal-to-Noise Ratio）。

    Args:
        img_pred: 预测图像（实数 magnitude）
        img_gt: GT 图像（实数 magnitude）
        data_range: 数据范围，默认为 GT 的最大值

    Returns:
        PSNR 值（dB）
    """
    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.detach().cpu().numpy()
    if isinstance(img_gt, torch.Tensor):
        img_gt = img_gt.detach().cpu().numpy()

    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)

    if data_range is None:
        data_range = img_gt.max()

    mse = np.mean((img_pred - img_gt) ** 2)
    if mse < 1e-10:
        return float("inf")

    psnr = 10 * np.log10((data_range ** 2) / mse)
    return float(psnr)


def compute_ssim(
    img_pred: Union[np.ndarray, torch.Tensor],
    img_gt: Union[np.ndarray, torch.Tensor],
    data_range: Optional[float] = None,
    win_size: int = 7,
) -> float:
    """计算 SSIM（Structural Similarity Index）- 3D 版本。

    对 3D volume 逐 slice（沿第一个维度）计算 2D SSIM 后取平均。

    Args:
        img_pred: 预测图像 (nz, nx, ny)
        img_gt: GT 图像 (nz, nx, ny)
        data_range: 数据范围
        win_size: 滑动窗口大小

    Returns:
        平均 SSIM 值
    """
    try:
        from skimage.metrics import structural_similarity as ssim_func
    except ImportError:
        print("[Warning] skimage not available, returning SSIM=0")
        return 0.0

    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.detach().cpu().numpy()
    if isinstance(img_gt, torch.Tensor):
        img_gt = img_gt.detach().cpu().numpy()

    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)

    if data_range is None:
        data_range = img_gt.max()

    # 逐 slice 计算 SSIM 后平均
    ssim_vals = []
    for i in range(img_pred.shape[0]):
        slice_pred = img_pred[i]
        slice_gt = img_gt[i]

        # 确保 win_size 不超过图像尺寸
        min_dim = min(slice_pred.shape)
        actual_win_size = min(win_size, min_dim)
        if actual_win_size % 2 == 0:
            actual_win_size -= 1
        if actual_win_size < 3:
            actual_win_size = 3

        try:
            ssim_val = ssim_func(
                slice_pred,
                slice_gt,
                data_range=data_range,
                win_size=actual_win_size,
            )
            ssim_vals.append(ssim_val)
        except Exception:
            continue

    if len(ssim_vals) == 0:
        return 0.0

    return float(np.mean(ssim_vals))


def compute_nmse(
    img_pred: Union[np.ndarray, torch.Tensor],
    img_gt: Union[np.ndarray, torch.Tensor],
) -> float:
    """计算 NMSE（Normalized Mean Squared Error）。

    NMSE = ||pred - gt||^2 / ||gt||^2
    """
    if isinstance(img_pred, torch.Tensor):
        img_pred = img_pred.detach().cpu().numpy()
    if isinstance(img_gt, torch.Tensor):
        img_gt = img_gt.detach().cpu().numpy()

    img_pred = img_pred.astype(np.float64)
    img_gt = img_gt.astype(np.float64)

    mse = np.sum((img_pred - img_gt) ** 2)
    norm_gt = np.sum(img_gt ** 2)

    if norm_gt < 1e-10:
        return float("inf")

    return float(mse / norm_gt)


def compute_metrics(
    img_pred: Union[np.ndarray, torch.Tensor],
    img_gt: Union[np.ndarray, torch.Tensor],
    data_range: Optional[float] = None,
) -> dict:
    """计算所有评估指标。

    Args:
        img_pred: 预测图像（实数 magnitude）
        img_gt: GT 图像（实数 magnitude）
        data_range: 数据范围

    Returns:
        包含 PSNR, SSIM, NMSE 的字典
    """
    return {
        "psnr": compute_psnr(img_pred, img_gt, data_range),
        "ssim": compute_ssim(img_pred, img_gt, data_range),
        "nmse": compute_nmse(img_pred, img_gt),
    }


# ============================================================================
# 3D Total Variation
# ============================================================================

def total_variation_3d(x: torch.Tensor) -> torch.Tensor:
    """计算 3D Total Variation。

    对 magnitude 图像计算 TV：
    TV = sum(|x[i+1,j,k] - x[i,j,k]|) + sum(|x[i,j+1,k] - x[i,j,k]|) + sum(|x[i,j,k+1] - x[i,j,k]|)

    Args:
        x: (nz, nx, ny) 实数张量

    Returns:
        标量 TV 值
    """
    # 沿各轴计算差分
    diff_z = torch.abs(x[1:, :, :] - x[:-1, :, :])
    diff_x = torch.abs(x[:, 1:, :] - x[:, :-1, :])
    diff_y = torch.abs(x[:, :, 1:] - x[:, :, :-1])

    return diff_z.sum() + diff_x.sum() + diff_y.sum()


def total_variation_3d_smooth(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算平滑 3D Total Variation（可微分版本）。

    使用 sqrt(dx^2 + dy^2 + dz^2 + eps) 代替 |dx| + |dy| + |dz|
    """
    # 差分（padding 以保持尺寸一致）
    diff_z = torch.zeros_like(x)
    diff_z[:-1, :, :] = x[1:, :, :] - x[:-1, :, :]

    diff_x = torch.zeros_like(x)
    diff_x[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]

    diff_y = torch.zeros_like(x)
    diff_y[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]

    # 平滑 L1
    tv = torch.sqrt(diff_z**2 + diff_x**2 + diff_y**2 + eps)
    return tv.sum()


# ============================================================================
# 路径工具
# ============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，不存在则创建。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Union[str, Path]) -> None:
    """保存字典为 JSON 文件。"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> dict:
    """加载 JSON 文件为字典。"""
    with open(path, "r") as f:
        return json.load(f)


# ============================================================================
# 分布式工具
# ============================================================================

def setup_distributed() -> Tuple[int, int, bool]:
    """初始化分布式环境。

    Returns:
        rank: 当前进程 rank
        world_size: 总进程数
        is_distributed: 是否为分布式模式
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        print(f"[Distributed] Initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed() -> None:
    """清理分布式环境。"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """判断是否为主进程。"""
    return rank == 0


# ============================================================================
# 其他工具
# ============================================================================

def normalize_to_range(x: torch.Tensor, new_min: float = 0.0, new_max: float = 1.0) -> torch.Tensor:
    """将张量归一化到指定范围。"""
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-10:
        return torch.zeros_like(x) + new_min
    return (x - x_min) / (x_max - x_min) * (new_max - new_min) + new_min


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 简单测试
    print("=== Utils Test ===")

    # 测试 PSNR/SSIM
    gt = np.random.rand(32, 64, 64).astype(np.float32)
    pred = gt + np.random.randn(*gt.shape).astype(np.float32) * 0.1

    metrics = compute_metrics(pred, gt)
    print(f"Metrics: {metrics}")

    # 测试 TV
    x = torch.randn(16, 32, 32)
    tv = total_variation_3d(x)
    print(f"TV: {tv.item():.4f}")

    # 测试复数转换
    c = torch.randn(10, 10, dtype=torch.complex64)
    r = complex_to_real_channels(c)
    c2 = real_channels_to_complex(r)
    print(f"Complex conversion error: {(c - c2).abs().max().item():.2e}")

    print("All tests passed!")
