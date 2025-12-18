#!/usr/bin/env python3
"""Training entry-point for the 3DGSMR reproduction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gsmr3d import (
    GaussianMRIModel,
    gaussian_undersampling_mask,
    load_kspace_volume,
    magnitude_tv_loss,
    psnr3d,
    ssim3d,
    fft3c,
    ifft3c,
)
from gsmr3d.gaussian_model import AdaptiveConfig


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3DGSMR training loop.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the .h5 file.")
    parser.add_argument("--acceleration", type=float, default=4.0, help="Undersampling factor.")
    parser.add_argument(
        "--center-fraction", type=float, default=0.08, help="Relative fully-sampled k-space cube."
    )
    parser.add_argument("--mask-seed", type=int, default=123, help="Random seed for the mask.")
    parser.add_argument("--device", type=str, default=default_device(), help="Training device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-gaussians", type=int, default=20000, help="Initial number of gaussians.")
    parser.add_argument("--mag-quantile", type=float, default=0.92, help="Quantile for initialization sampling.")
    parser.add_argument(
        "--scale-voxels",
        type=float,
        nargs=3,
        default=(3.0, 3.0, 3.0),
        help="Initial gaussian std (in voxels) along (z, y, x).",
    )
    parser.add_argument("--max-gaussians", type=int, default=400000, help="Upper bound during densification.")
    parser.add_argument("--max-iters", type=int, default=600, help="Maximum number of iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--tv-weight", type=float, default=5e-5, help="Weight of the magnitude TV penalty.")
    parser.add_argument("--densify-interval", type=int, default=100, help="How often to densify/adapt.")
    parser.add_argument("--grad-threshold", type=float, default=0.15, help="Gradient magnitude split threshold.")
    parser.add_argument("--amp-threshold", type=float, default=5e-4, help="Pruning threshold on |rho|.")
    parser.add_argument("--grad-floor", type=float, default=0.05, help="Minimum grad during pruning.")
    parser.add_argument("--max-add-per-interval", type=int, default=1024, help="Maximum new points after splitting.")
    parser.add_argument("--log-interval", type=int, default=25, help="Iterations between console logs.")
    parser.add_argument(
        "--save-dir", type=Path, default=Path("outputs"), help="Directory where reconstructions will be saved."
    )
    parser.add_argument(
        "--combine-coils",
        action="store_true",
        help="Sum k-space across the coil dimension (default True).",
    )
    parser.add_argument("--no-combine-coils", dest="combine_coils", action="store_false")
    parser.set_defaults(combine_coils=True)
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Optional periodic checkpointing.")
    return parser


def save_volume(volume: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(volume.detach().cpu(), path)


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    kspace = load_kspace_volume(args.data, combine_coils=args.combine_coils).to(device)
    shape = kspace.shape

    mask = gaussian_undersampling_mask(
        shape,
        acceleration=args.acceleration,
        center_fraction=args.center_fraction,
        seed=args.mask_seed,
        device=device,
    )
    undersampled_kspace = kspace * mask

    fully_sampled = ifft3c(kspace)
    aliased = ifft3c(undersampled_kspace)

    adaptive_cfg = AdaptiveConfig(
        densify_interval=args.densify_interval,
        gradient_threshold=args.grad_threshold,
        amplitude_threshold=args.amp_threshold,
        max_add_per_interval=args.max_add_per_interval,
    )

    # 找到这段代码 (大约 123行)
    model = GaussianMRIModel.from_initial_volume(
        aliased,
        num_points=args.num_gaussians, # 确保这里的 args.num_gaussians 是 500
        magnitude_quantile=args.mag_quantile,
        # scale_voxels=tuple(args.scale_voxels),  <-- 删除这一行，因为现在是自动计算的
        max_points=args.max_gaussians,
        adaptive=adaptive_cfg,
        seed=args.seed,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with open(save_dir / "config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)

    print(f"Loaded volume shape: {shape}, initial gaussians: {model.num_points}")
    print(f"Mask keeps {mask.sum().item():.0f} / {mask.numel()} samples.")

    start = time.time()
    best_psnr = -1.0
    best_volume = None

    for it in range(1, args.max_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        recon = model()
        pred_kspace = fft3c(recon)
        diff = pred_kspace * mask - undersampled_kspace
        data_consistency = (diff.abs() ** 2).mean()
        loss = data_consistency

        if args.tv_weight > 0:
            loss = loss + args.tv_weight * magnitude_tv_loss(recon)

        loss.backward()
        center_grad_norm = model.centers.grad.detach().norm(dim=-1)
        optimizer.step()

        if args.densify_interval > 0 and it % args.densify_interval == 0:
            with torch.no_grad():
                added = model.long_axis_split(center_grad_norm, args.max_add_per_interval)
                pruned = model.prune(args.amp_threshold, args.grad_floor, center_grad_norm)

            if added or pruned:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if it % args.log_interval == 0 or it == args.max_iters:
            with torch.no_grad():
                psnr_val = psnr3d(recon, fully_sampled)
                ssim_val = ssim3d(recon, fully_sampled)
                elapsed = time.time() - start
                print(
                    f"[Iter {it:04d}] loss={loss.item():.6f} PSNR={psnr_val:.2f}dB SSIM={ssim_val:.3f} "
                    f"Gaussians={model.num_points} time={elapsed:.1f}s"
                )
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    best_volume = recon.detach().clone()

        if args.checkpoint_every and it % args.checkpoint_every == 0:
            save_volume(recon, save_dir / f"reconstruction_iter_{it:04d}.pt")

    if best_volume is None:
        best_volume = recon.detach()

    save_volume(best_volume, save_dir / "reconstruction_final.pt")
    save_volume(mask, save_dir / "mask.pt")
    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB. Outputs stored in {save_dir}.")


if __name__ == "__main__":
    main(build_argparser().parse_args())
