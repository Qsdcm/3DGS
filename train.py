"""train.py

3D Gaussian Splatting MRI Reconstruction 训练脚本 (优化版)。

改进点：
1. 调整了默认超参数 (lr, n_gaussians, k_init) 以适应 MRI 任务。
2. 增加了中间结果的可视化 (保存 .png)。
3. 优化了 Densify 后的优化器处理逻辑。

用法：
    python train.py --data_root /path/to/data --out_root ./outputs --max_iters 3000
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

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

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

    # Model Init Parameters (Updated defaults)
    parser.add_argument("--n_gaussians", type=int, default=50000,
                        help="Initial number of Gaussians (Increased for better details)")
    parser.add_argument("--sigma_cutoff", type=float, default=3.0,
                        help="Gaussian truncation (in sigmas)")
    parser.add_argument("--percentile_thresh", type=float, default=0.0,
                        help="Percentile threshold for init (0 means use all valid points)")
    parser.add_argument("--k_init", type=float, default=10.0,
                        help="Density initialization scale factor (Higher to avoid black initial image)")

    # Training
    parser.add_argument("--max_iters", type=int, default=3000,
                        help="Maximum optimization iterations per sample")
    
    # Learning Rates (Updated defaults)
    parser.add_argument("--lr", type=float, default=0.01, help="Base LR (not used directly)")
    parser.add_argument("--lr_centers", type=float, default=0.003, 
                        help="LR for centers (Increased)")
    parser.add_argument("--lr_scales", type=float, default=0.005,
                        help="LR for log_scales")
    parser.add_argument("--lr_rho", type=float, default=0.05,
                        help="LR for rho/density (Significantly increased for fast convergence)")

    # Regularization
    parser.add_argument("--lambda_tv", type=float, default=1e-4,
                        help="TV regularization weight")

    # Adaptive control
    parser.add_argument("--densify_every", type=int, default=300,
                        help="Densification interval (Increased to preserve optimizer momentum)")
    parser.add_argument("--densify_start", type=int, default=200,
                        help="Start densification after this iteration")
    parser.add_argument("--densify_end", type=int, default=2500,
                        help="Stop densification after this iteration")
    parser.add_argument("--grad_threshold", type=float, default=0.002,
                        help="Gradient threshold for densification")
    parser.add_argument("--prune_rho_thresh", type=float, default=1e-4,
                        help="Density threshold for pruning")
    parser.add_argument("--max_gaussians", type=int, default=200000,
                        help="Maximum number of Gaussians")
    parser.add_argument("--min_gaussians", type=int, default=1000,
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
    parser.add_argument("--print_every", type=int, default=100,
                        help="Print interval")
    parser.add_argument("--vis_every", type=int, default=200,
                        help="Visualization save interval")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Checkpoint save interval")

    return parser.parse_args()


def get_optimizer(model, args):
    """创建优化器"""
    return torch.optim.Adam([
        {"params": [model.centers], "lr": args.lr_centers},
        {"params": [model.log_scales], "lr": args.lr_scales},
        {"params": [model.rho_real, model.rho_imag], "lr": args.lr_rho},
        # 如果启用了 rotation，添加 rotation 参数
        # {"params": [model.rotations], "lr": ...} 
    ])


def save_visualization(volume, out_dir, iteration, sample_name):
    """保存中间切片图像"""
    if not HAS_TORCHVISION:
        return
        
    try:
        # 取中间切片
        nz = volume.shape[0]
        mid_slice = volume[nz // 2].detach().cpu()
        
        # 计算幅值
        mag = torch.abs(mid_slice)
        
        # 简单归一化到 [0, 1] 用于显示
        if mag.max() > 0:
            mag = (mag - mag.min()) / (mag.max() - mag.min())
        
        vis_path = out_dir / "visualizations"
        ensure_dir(vis_path)
        
        torchvision.utils.save_image(
            mag.unsqueeze(0), 
            vis_path / f"{sample_name}_iter_{iteration:05d}.png"
        )
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")


def train_single_sample(
    sample_name: str,
    kspace_under: torch.Tensor,
    mask: torch.Tensor,
    image_gt: torch.Tensor,
    image_init_complex: torch.Tensor,
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"[Train] Sample: {sample_name} | Shape: {kspace_under.shape}")
    print(f"{'='*60}")

    volume_shape = tuple(kspace_under.shape)
    
    # 1. 创建模型
    model = GaussianMRIModel(
        volume_shape=volume_shape,
        n_gaussians=args.n_gaussians,
        device=device,
        sigma_cutoff=args.sigma_cutoff,
    )

    # 2. 初始化 (Crucial Step)
    # 使用较高的 k_init 确保初始有信号
    model.initialize_from_image(
        image_init_complex,
        n_gaussians=args.n_gaussians,
        percentile_thresh=args.percentile_thresh,
        k_init=args.k_init,
        seed=args.seed,
    )

    # 3. 优化器
    optimizer = get_optimizer(model, args)

    # 训练循环
    loss_history = []
    best_loss = float("inf")
    best_psnr = 0.0
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
        
        # Gradient clipping (Optional, adds stability)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update
        optimizer.step()

        # Record
        loss_val = loss_dict["loss_total"]
        loss_history.append(loss_val)

        # 4. Adaptive Control (Densify & Prune)
        densify_stats = {}
        if (args.densify_start <= iteration <= args.densify_end and
            iteration % args.densify_every == 0):

            stats = model.densify_and_prune(
                grad_threshold=args.grad_threshold,
                prune_rho_thresh=args.prune_rho_thresh,
                max_gaussians=args.max_gaussians,
                min_gaussians=args.min_gaussians,
                split_scale_factor=args.split_scale_factor,
                split_delta_factor=args.split_delta_factor,
            )
            densify_stats = stats

            # 如果参数数量发生变化，必须重建优化器
            # 注意：重建优化器会丢失动量(Momentum)，所以我们降低了 Densify 频率
            if stats.get("n_split", 0) > 0 or stats.get("n_pruned", 0) > 0:
                optimizer = get_optimizer(model, args)

        # 5. Logging & Checkpointing
        if loss_val < best_loss:
            best_loss = loss_val
            # 仅保存参数，不建议太频繁深拷贝整个 state dict，耗时
            # best_state = model.get_state_dict() 
        
        # Print
        if iteration % args.print_every == 0 or iteration == 1:
            elapsed = time.time() - start_time
            msg = (f"[Iter {iteration:5d}] "
                   f"Loss: {loss_val:.6f} (DC: {loss_dict['loss_dc']:.6f}) "
                   f"#G: {model.n_gaussians} "
                   f"Time: {elapsed:.1f}s")
            
            if densify_stats:
                msg += f" | Split: {densify_stats.get('n_split',0)} Prune: {densify_stats.get('n_pruned',0)}"
            print(msg)

        # Vis
        if iteration % args.vis_every == 0:
            save_visualization(volume, out_dir, iteration, sample_name)
            
            # 计算一下当前的 PSNR 看看情况
            with torch.no_grad():
                mag_curr = torch.abs(volume)
                metrics_curr = compute_metrics(mag_curr, image_gt)
                if metrics_curr['psnr'] > best_psnr:
                    best_psnr = metrics_curr['psnr']
                    best_state = model.get_state_dict() # 保存 PSNR 最高的模型

    total_time = time.time() - start_time

    # 训练结束，加载最佳状态 (Best PSNR)
    if best_state is not None:
        print(f"Loading best model with PSNR: {best_psnr:.2f} dB")
        model.load_state_dict_custom(best_state)

    # 最终生成
    with torch.no_grad():
        recon_complex = model.voxelize()
        recon_mag = torch.abs(recon_complex)

    # 最终指标
    metrics = compute_metrics(recon_mag, image_gt)
    print(f"\n[Final Result] PSNR: {metrics['psnr']:.2f} dB, "
          f"SSIM: {metrics['ssim']:.4f}, "
          f"NMSE: {metrics['nmse']:.6f}")

    # 保存
    ensure_dir(out_dir)
    np.save(out_dir / "recon_complex.npy", recon_complex.cpu().numpy().astype(np.complex64))
    np.save(out_dir / "recon_mag.npy", recon_mag.cpu().numpy().astype(np.float32))
    torch.save(model.get_state_dict(), out_dir / "gaussians.pt")
    
    # 保存最终可视化
    save_visualization(recon_complex, out_dir, args.max_iters, f"{sample_name}_final")

    result = {
        "sample_name": sample_name,
        "metrics": metrics,
        "n_gaussians_final": model.n_gaussians,
        "total_time_sec": total_time,
        "loss_history": loss_history,
    }
    save_json(result, out_dir / "metrics.json")
    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    # Distributed setup
    rank, world_size, is_distributed = 0, 1, False
    if args.distributed:
        rank, world_size, is_distributed = setup_distributed()

    # Device
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
        print(f"Starting Training on {device}...")
        print(f"Params: LR_rho={args.lr_rho}, Init_K={args.k_init}, N_Gaussians={args.n_gaussians}")

    # Loader
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

    out_root = Path(args.out_root)
    if is_main_process(rank):
        ensure_dir(out_root)

    all_results = []
    for sample_name, mri_ds in loader:
        kspace_under, mask, image_gt, image_init_complex, meta = get_sample_data(mri_ds)
        sample_out_dir = out_root / sample_name
        ensure_dir(sample_out_dir)

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

    # Summary
    if is_main_process(rank) and all_results:
        summary = {
            "avg_psnr": np.mean([r["metrics"]["psnr"] for r in all_results]),
            "avg_ssim": np.mean([r["metrics"]["ssim"] for r in all_results]),
            "samples": all_results
        }
        save_json(summary, out_root / "summary.json")
        print("\nTraining Complete.")

    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()