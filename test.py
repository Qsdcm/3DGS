"""test.py

3D Gaussian Splatting MRI Reconstruction 测试/评估脚本。

功能：
1. 读取 out_root 下的重建结果
2. 如果提供 data_root，重新加载 GT 进行对比
3. 计算/汇总评估指标
4. 输出 results.json

用法：
    python test.py --out_root ./outputs
    python test.py --data_root /path/to/data --out_root ./outputs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from utils import (
    compute_metrics,
    save_json,
    load_json,
    ensure_dir,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3DGS MRI Reconstruction Testing/Evaluation")

    parser.add_argument("--out_root", type=str, required=True,
                        help="Output directory containing reconstruction results")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to H5 file or directory (optional, for recomputing GT comparison)")

    # Data preprocessing (only used if data_root is provided)
    parser.add_argument("--acceleration", type=float, default=4.0,
                        help="Undersampling acceleration factor")
    parser.add_argument("--mask_type", type=str, default="gaussian",
                        choices=["gaussian", "uniform"],
                        help="Mask type for undersampling")
    parser.add_argument("--center_fraction", type=float, default=0.08,
                        help="Center fraction for ACS lines")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for mask generation)")

    # Distributed
    parser.add_argument("--distributed", type=int, default=0,
                        help="Enable distributed testing (0 or 1)")

    # Output
    parser.add_argument("--output_file", type=str, default="results.json",
                        help="Output results filename")

    return parser.parse_args()


def find_sample_dirs(out_root: Path) -> List[Path]:
    """查找所有包含重建结果的样本目录。"""
    sample_dirs = []

    for item in out_root.iterdir():
        if item.is_dir():
            # 检查是否包含必要的文件
            if (item / "recon_mag.npy").exists() or (item / "metrics.json").exists():
                sample_dirs.append(item)

    return sorted(sample_dirs)


def evaluate_sample(
    sample_dir: Path,
    gt_image: Optional[torch.Tensor] = None,
    recompute: bool = False,
) -> Dict:
    """评估单个样本。

    Args:
        sample_dir: 样本结果目录
        gt_image: GT 图像（可选，如果提供则重新计算指标）
        recompute: 是否强制重新计算

    Returns:
        result: 评估结果字典
    """
    sample_name = sample_dir.name
    result = {"sample_name": sample_name}

    # 尝试加载已有的 metrics
    metrics_file = sample_dir / "metrics.json"
    if metrics_file.exists() and not recompute:
        try:
            saved_metrics = load_json(metrics_file)
            result["metrics"] = saved_metrics.get("metrics", {})
            result["n_gaussians_final"] = saved_metrics.get("n_gaussians_final", None)
            result["total_time_sec"] = saved_metrics.get("total_time_sec", None)
            result["source"] = "saved"
        except Exception as e:
            print(f"[Warning] Failed to load metrics for {sample_name}: {e}")
            result["metrics"] = {}
            result["source"] = "error"

    # 如果提供了 GT，重新计算指标
    if gt_image is not None:
        recon_mag_file = sample_dir / "recon_mag.npy"
        if recon_mag_file.exists():
            try:
                recon_mag = np.load(recon_mag_file)

                # 确保形状匹配
                if recon_mag.shape != tuple(gt_image.shape):
                    print(f"[Warning] Shape mismatch for {sample_name}: "
                          f"recon {recon_mag.shape} vs gt {tuple(gt_image.shape)}")
                else:
                    # 转换为 numpy
                    if isinstance(gt_image, torch.Tensor):
                        gt_np = gt_image.cpu().numpy()
                    else:
                        gt_np = gt_image

                    metrics = compute_metrics(recon_mag, gt_np)
                    result["metrics"] = metrics
                    result["source"] = "recomputed"

            except Exception as e:
                print(f"[Warning] Failed to recompute metrics for {sample_name}: {e}")

    return result


def main():
    args = parse_args()

    # 分布式初始化
    rank, world_size, is_distributed = 0, 1, False
    if args.distributed:
        rank, world_size, is_distributed = setup_distributed()

    out_root = Path(args.out_root)

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"3D Gaussian Splatting MRI Reconstruction - Evaluation")
        print(f"{'='*60}")
        print(f"Output root: {out_root}")
        print(f"Data root: {args.data_root or 'Not provided'}")
        print(f"{'='*60}\n")

    # 查找样本目录
    sample_dirs = find_sample_dirs(out_root)

    if len(sample_dirs) == 0:
        print(f"[Error] No sample results found in {out_root}")
        sys.exit(1)

    if is_main_process(rank):
        print(f"Found {len(sample_dirs)} samples")

    # 分布式切分
    if is_distributed:
        sample_dirs = sample_dirs[rank::world_size]
        print(f"[Rank {rank}] Processing {len(sample_dirs)} samples")

    # 加载数据（如果提供）
    gt_images = {}
    if args.data_root is not None:
        try:
            from dataset import MRIDatasetLoader, get_sample_data

            loader = MRIDatasetLoader(
                data_root=args.data_root,
                acceleration=args.acceleration,
                center_fraction=args.center_fraction,
                mask_type=args.mask_type,
                seed=args.seed,
                device="cpu",  # 评估在 CPU 上进行
                rank=0,
                world_size=1,  # 加载所有数据
            )

            for sample_name, mri_ds in loader:
                _, _, image_gt, _, _ = get_sample_data(mri_ds)
                gt_images[sample_name] = image_gt

            print(f"Loaded GT images for {len(gt_images)} samples")

        except Exception as e:
            print(f"[Warning] Failed to load GT data: {e}")
            gt_images = {}

    # 评估每个样本
    all_results = []

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        gt_image = gt_images.get(sample_name, None)

        result = evaluate_sample(
            sample_dir=sample_dir,
            gt_image=gt_image,
            recompute=(gt_image is not None),
        )

        all_results.append(result)

        if is_main_process(rank):
            metrics = result.get("metrics", {})
            psnr = metrics.get("psnr", "N/A")
            ssim = metrics.get("ssim", "N/A")
            source = result.get("source", "unknown")

            if isinstance(psnr, float):
                print(f"  {sample_name}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f} ({source})")
            else:
                print(f"  {sample_name}: metrics unavailable")

    # 汇总结果
    if is_distributed:
        # 收集所有 rank 的结果
        if torch.distributed.is_initialized():
            # 简单实现：每个 rank 保存自己的部分结果
            partial_results_file = out_root / f"partial_results_rank{rank}.json"
            save_json(all_results, partial_results_file)

            torch.distributed.barrier()

            if is_main_process(rank):
                # 主进程合并所有结果
                all_results = []
                for r in range(world_size):
                    partial_file = out_root / f"partial_results_rank{r}.json"
                    if partial_file.exists():
                        partial = load_json(partial_file)
                        all_results.extend(partial)
                        partial_file.unlink()  # 删除临时文件

    # 计算汇总统计（仅主进程）
    if is_main_process(rank):
        # 过滤有效结果
        valid_results = [r for r in all_results if r.get("metrics", {}).get("psnr") is not None]

        if len(valid_results) > 0:
            psnr_values = [r["metrics"]["psnr"] for r in valid_results]
            ssim_values = [r["metrics"]["ssim"] for r in valid_results]
            nmse_values = [r["metrics"].get("nmse", float("nan")) for r in valid_results]

            summary = {
                "n_samples_total": len(all_results),
                "n_samples_valid": len(valid_results),
                "psnr": {
                    "mean": float(np.mean(psnr_values)),
                    "std": float(np.std(psnr_values)),
                    "min": float(np.min(psnr_values)),
                    "max": float(np.max(psnr_values)),
                },
                "ssim": {
                    "mean": float(np.mean(ssim_values)),
                    "std": float(np.std(ssim_values)),
                    "min": float(np.min(ssim_values)),
                    "max": float(np.max(ssim_values)),
                },
                "nmse": {
                    "mean": float(np.nanmean(nmse_values)),
                    "std": float(np.nanstd(nmse_values)),
                },
                "samples": [
                    {
                        "name": r["sample_name"],
                        "psnr": r["metrics"].get("psnr"),
                        "ssim": r["metrics"].get("ssim"),
                        "nmse": r["metrics"].get("nmse"),
                        "source": r.get("source"),
                    }
                    for r in all_results
                ],
            }
        else:
            summary = {
                "n_samples_total": len(all_results),
                "n_samples_valid": 0,
                "error": "No valid metrics found",
                "samples": [{"name": r["sample_name"]} for r in all_results],
            }

        # 保存结果
        results_file = out_root / args.output_file
        save_json(summary, results_file)

        # 打印汇总
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"Total samples: {summary['n_samples_total']}")
        print(f"Valid samples: {summary['n_samples_valid']}")

        if summary['n_samples_valid'] > 0:
            print(f"\nPSNR (dB):")
            print(f"  Mean: {summary['psnr']['mean']:.2f} ± {summary['psnr']['std']:.2f}")
            print(f"  Range: [{summary['psnr']['min']:.2f}, {summary['psnr']['max']:.2f}]")
            print(f"\nSSIM:")
            print(f"  Mean: {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
            print(f"  Range: [{summary['ssim']['min']:.4f}, {summary['ssim']['max']:.4f}]")
            print(f"\nNMSE:")
            print(f"  Mean: {summary['nmse']['mean']:.6f} ± {summary['nmse']['std']:.6f}")

        print(f"\nResults saved to: {results_file}")
        print(f"{'='*60}\n")

    # 清理分布式
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
