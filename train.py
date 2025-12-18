"""train.py

3D Gaussian Splatting MRI Reconstruction 训练脚本。

这是"每个 volume 的优化式重建"：对数据集中的每个 H5 样本执行 Adam 优化。

输出（每个样本）：
- recon_complex.npy: 重建的复数图像
- recon_mag.npy: 重建的幅值图像
- gaussians.pt: 最终 Gaussian 参数
- metrics.json: 评估指标和损失曲线

支持：
- 单 GPU: python train.py --data_root ...
- 多 GPU: torchrun --nproc_per_node=N train.py --data_root ... --distributed 1

用法：
    python train.py --data_root /path/to/data --out_root ./outputs --max_iters 2000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset import MRIDatasetLoader, get_sample_data
from model import GaussianMRIModel
from utils import (
    set_seed,
    compute_metrics,
    ensure_dir,
    save_json,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3DGS MRI Reconstruction Training")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to H5 file or directory containing H5 files")
    parser.add_argument("--out_root", type=str, default="./outputs",
                        help="Output directory")

    # Data preprocessing
    parser.add_argument("--acceleration", type=float, default=4.0,
                        help="Undersampling acceleration factor")
    parser.add_argument("--mask_type", type=str, default="gaussian",
                        choices=["gaussian", "uniform"],
                        help="Mask type for undersampling")
    parser.add_argument("--center_fraction", type=float, default=0.08,
                        help="Center fraction for ACS lines")

    # Model
    parser.add_argument("--n_gaussians", type=int, default=10000,
                        help="Initial number of Gaussians")
    parser.add_argument("--sigma_cutoff", type=float, default=3.0,
                        help="Gaussian truncation (in sigmas)")
    parser.add_argument("--percentile_thresh", type=float, default=10.0,
                        help="Percentile threshold for background filtering in initialization")
    parser.add_argument("--k_init", type=float, default=1.0,
                        help="Density initialization scale factor")

    # Training
    parser.add_argument("--max_iters", type=int, default=5000,
                        help="Maximum optimization iterations per sample")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--lr_centers", type=float, default=0.001,
                        help="Learning rate for centers")
    parser.add_argument("--lr_scales", type=float, default=0.005,
                        help="Learning rate for log_scales")
    parser.add_argument("--lr_rho", type=float, default=0.01,
                        help="Learning rate for rho (density)")

    # Regularization
    parser.add_argument("--lambda_tv", type=float, default=1e-4,
                        help="TV regularization weight (0 to disable)")

    # Adaptive control
    parser.add_argument("--densify_every", type=int, default=100,
                        help="Densification interval (iterations)")
    parser.add_argument("--densify_start", type=int, default=50,
                        help="Start densification after this iteration")
    parser.add_argument("--densify_end", type=int, default=2000,
                        help="Stop densification after this iteration")
    parser.add_argument("--grad_threshold", type=float, default=0.005,
                        help="Gradient threshold for densification")
    parser.add_argument("--prune_rho_thresh", type=float, default=1e-4,
                        help="Density threshold for pruning")
    parser.add_argument("--max_gaussians", type=int, default=100000,
                        help="Maximum number of Gaussians")
    parser.add_argument("--min_gaussians", type=int, default=64,
                        help="Minimum Gaussians kept after pruning")
    parser.add_argument("--split_scale_factor", type=float, default=0.7,
                        help="Scale reduction factor after split")
    parser.add_argument("--split_delta_factor", type=float, default=0.5,
                        help="Position offset factor for split")

    # Distributed
    parser.add_argument("--distributed", type=int, default=0,
                        help="Enable distributed training (0 or 1)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print interval")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Checkpoint save interval (0 to disable)")

    return parser.parse_args()


def train_single_sample(
    sample_name: str,
    kspace_under: torch.Tensor,  # (nz, nx, ny) complex
    mask: torch.Tensor,          # (nz, nx, ny) float
    image_gt: torch.Tensor,      # (nz, nx, ny) float
    image_init_complex: torch.Tensor,  # (nz, nx, ny) complex
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict:
    """对单个样本执行优化式重建。

    Returns:
        result: 包含 metrics 和 timing 的字典
    """
    print(f"\n{'='*60}")
    print(f"[Train] Sample: {sample_name}")
    print(f"{'='*60}")

    volume_shape = tuple(kspace_under.shape)
    nz, nx, ny = volume_shape

    # 创建模型
    model = GaussianMRIModel(
        volume_shape=volume_shape,
        n_gaussians=args.n_gaussians,
        device=device,
        sigma_cutoff=args.sigma_cutoff,
    )

    # 从初始图像初始化
    model.initialize_from_image(
        image_init_complex,
        n_gaussians=args.n_gaussians,
        percentile_thresh=args.percentile_thresh,
        k_init=args.k_init,
        seed=args.seed,
    )

    # 设置优化器（分组学习率）
    optimizer = torch.optim.Adam([
        {"params": [model.centers], "lr": args.lr_centers},
        {"params": [model.log_scales], "lr": args.lr_scales},
        {"params": [model.rho_real, model.rho_imag], "lr": args.lr_rho},
    ])

    # 训练循环
    loss_history = []
    best_loss = float("inf")
    best_state = None

    start_time = time.time()

    for iteration in range(1, args.max_iters + 1):
        optimizer.zero_grad()

        # Forward
        volume, kspace_pred = model.forward(mask)

        # Loss
        loss, loss_dict = model.compute_loss(
            kspace_pred,
            kspace_under,
            mask,
            volume=volume,
            lambda_tv=args.lambda_tv,
        )

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Record
        loss_history.append(loss_dict["loss_total"])

        if loss_dict["loss_total"] < best_loss:
            best_loss = loss_dict["loss_total"]
            best_state = model.get_state_dict()

        # Densification & Pruning
        densify_stats = None
        if (args.densify_start <= iteration <= args.densify_end and
            iteration % args.densify_every == 0):

            densify_stats = model.densify_and_prune(
                grad_threshold=args.grad_threshold,
                prune_rho_thresh=args.prune_rho_thresh,
                max_gaussians=args.max_gaussians,
                min_gaussians=args.min_gaussians,
                split_scale_factor=args.split_scale_factor,
                split_delta_factor=args.split_delta_factor,
            )

            # 重新创建优化器（参数数量可能变化）
            optimizer = torch.optim.Adam([
                {"params": [model.centers], "lr": args.lr_centers},
                {"params": [model.log_scales], "lr": args.lr_scales},
                {"params": [model.rho_real, model.rho_imag], "lr": args.lr_rho},
            ])

        # Print progress
        if iteration % args.print_every == 0 or iteration == 1:
            elapsed = time.time() - start_time
            msg = (f"[Iter {iteration:5d}/{args.max_iters}] "
                   f"Loss: {loss_dict['loss_total']:.6f} "
                   f"(DC: {loss_dict['loss_dc']:.6f}) "
                   f"#G: {model.n_gaussians:5d} "
                   f"Time: {elapsed:.1f}s")
            if densify_stats:
                msg += f" | Split: {densify_stats['n_split']}, Prune: {densify_stats['n_pruned']}"
            print(msg)

        # Save checkpoint
        if args.save_every > 0 and iteration % args.save_every == 0:
            ckpt_path = out_dir / f"checkpoint_{iteration:05d}.pt"
            torch.save(model.get_state_dict(), ckpt_path)

    total_time = time.time() - start_time

    # 恢复最佳状态
    if best_state is not None:
        model.load_state_dict_custom(best_state)

    # 生成最终重建
    with torch.no_grad():
        recon_complex = model.voxelize()
        recon_mag = torch.abs(recon_complex)

    # 计算评估指标
    metrics = compute_metrics(
        recon_mag.cpu(),
        image_gt.cpu(),
    )

    print(f"\n[Result] PSNR: {metrics['psnr']:.2f} dB, "
          f"SSIM: {metrics['ssim']:.4f}, "
          f"NMSE: {metrics['nmse']:.6f}")

    # 保存结果
    ensure_dir(out_dir)

    # 复数重建
    recon_complex_np = recon_complex.cpu().numpy().astype(np.complex64)
    np.save(out_dir / "recon_complex.npy", recon_complex_np)

    # 幅值重建
    recon_mag_np = recon_mag.cpu().numpy().astype(np.float32)
    np.save(out_dir / "recon_mag.npy", recon_mag_np)

    # Gaussian 参数
    torch.save(model.get_state_dict(), out_dir / "gaussians.pt")

    # Metrics
    result = {
        "sample_name": sample_name,
        "metrics": metrics,
        "final_loss": loss_history[-1] if loss_history else None,
        "best_loss": best_loss,
        "n_gaussians_final": model.n_gaussians,
        "total_time_sec": total_time,
        "iterations": args.max_iters,
        "loss_history": loss_history,
    }
    save_json(result, out_dir / "metrics.json")

    print(f"[Train] Results saved to: {out_dir}")

    return result


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 分布式初始化
    rank, world_size, is_distributed = 0, 1, False
    if args.distributed:
        rank, world_size, is_distributed = setup_distributed()

    # 设备
    if torch.cuda.is_available():
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
        print("[Warning] CUDA not available, using CPU")

    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"3D Gaussian Splatting MRI Reconstruction")
        print(f"{'='*60}")
        print(f"Data root: {args.data_root}")
        print(f"Output root: {args.out_root}")
        print(f"Device: {device}")
        print(f"Distributed: {is_distributed} (rank {rank}/{world_size})")
        print(f"Max iterations: {args.max_iters}")
        print(f"Initial Gaussians: {args.n_gaussians}")
        print(f"Acceleration: {args.acceleration}x")
        print(f"Mask type: {args.mask_type}")
        print(f"{'='*60}\n")

    # 创建数据加载器
    loader = MRIDatasetLoader(
        data_root=args.data_root,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        mask_type=args.mask_type,
        seed=args.seed,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    if len(loader) == 0:
        print(f"[Error] No samples found in {args.data_root}")
        sys.exit(1)

    # 创建输出目录
    out_root = Path(args.out_root)
    if is_main_process(rank):
        ensure_dir(out_root)

    # 遍历样本进行重建
    all_results = []

    for sample_name, mri_ds in loader:
        # 获取数据
        kspace_under, mask, image_gt, image_init_complex, meta = get_sample_data(mri_ds)

        # 样本输出目录
        sample_out_dir = out_root / sample_name

        # 训练
        result = train_single_sample(
            sample_name=sample_name,
            kspace_under=kspace_under,
            mask=mask,
            image_gt=image_gt,
            image_init_complex=image_init_complex,
            out_dir=sample_out_dir,
            args=args,
            device=device,
        )

        all_results.append(result)

    # 汇总结果（仅 rank 0）
    if is_main_process(rank) and len(all_results) > 0:
        summary = {
            "n_samples": len(all_results),
            "avg_psnr": np.mean([r["metrics"]["psnr"] for r in all_results]),
            "avg_ssim": np.mean([r["metrics"]["ssim"] for r in all_results]),
            "avg_nmse": np.mean([r["metrics"]["nmse"] for r in all_results]),
            "total_time_sec": sum([r["total_time_sec"] for r in all_results]),
            "samples": [
                {
                    "name": r["sample_name"],
                    "psnr": r["metrics"]["psnr"],
                    "ssim": r["metrics"]["ssim"],
                    "nmse": r["metrics"]["nmse"],
                }
                for r in all_results
            ],
        }
        save_json(summary, out_root / "summary.json")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Samples processed: {summary['n_samples']}")
        print(f"Average PSNR: {summary['avg_psnr']:.2f} dB")
        print(f"Average SSIM: {summary['avg_ssim']:.4f}")
        print(f"Average NMSE: {summary['avg_nmse']:.6f}")
        print(f"Total time: {summary['total_time_sec']:.1f}s")
        print(f"Results saved to: {out_root}")
        print(f"{'='*60}\n")

    # 清理分布式
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
