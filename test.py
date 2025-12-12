"""
3DGSMR Test/Inference Script
加载训练好的模型进行推理和评估
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import scipy.io as scio
from datetime import datetime

from dataset import (
    MRIDataset, 
    kspace_to_image, 
    image_to_kspace, 
    coil_combine_sos
)
from model import (
    ComplexGaussianModel, 
    Voxelizer, 
    VoxelizerOptimized,
    compute_psnr,
    compute_ssim
)


class Tester:
    """
    3DGSMR 测试器
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:1'):
        """
        初始化测试器
        
        Args:
            checkpoint_path: 检查点文件路径
            device: 目标设备
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # 加载检查点
        self._load_checkpoint()
        
        # 初始化数据集
        self._init_dataset()
        
        # 初始化模型
        self._init_model()
    
    def _load_checkpoint(self):
        """加载检查点"""
        print(f"\nLoading checkpoint from: {self.checkpoint_path}")
        
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.config = self.checkpoint['config']
        
        print(f"Checkpoint iteration: {self.checkpoint['iteration']}")
        print(f"Number of Gaussians: {self.checkpoint['model_state']['num_gaussians']}")
    
    def _init_dataset(self):
        """初始化数据集"""
        print("\nInitializing Dataset...")
        
        # 使用与训练相同的配置
        self.dataset = MRIDataset(
            h5_path=self.config['data_path'],
            acceleration=self.config.get('acceleration', 4.0),
            center_fraction=self.config.get('center_fraction', 0.08),
            mask_type=self.config.get('mask_type', 'gaussian'),
            seed=self.config.get('seed', 42),
            device=self.device
        )
        
        self.volume_shape = self.dataset.get_volume_shape()
        print(f"Volume shape: {self.volume_shape}")
    
    def _init_model(self):
        """初始化并恢复模型"""
        print("\nInitializing Model...")
        
        model_state = self.checkpoint['model_state']
        num_gaussians = model_state['num_gaussians']
        
        # 创建模型
        self.gaussian_model = ComplexGaussianModel(
            num_gaussians=num_gaussians,
            volume_shape=self.volume_shape,
            device=self.device
        )
        
        # 恢复参数
        self.gaussian_model.positions = nn.Parameter(
            model_state['positions'].to(self.device)
        )
        self.gaussian_model.log_scales = nn.Parameter(
            model_state['log_scales'].to(self.device)
        )
        self.gaussian_model.rotations = nn.Parameter(
            model_state['rotations'].to(self.device)
        )
        self.gaussian_model.densities_real = nn.Parameter(
            model_state['densities_real'].to(self.device)
        )
        self.gaussian_model.densities_imag = nn.Parameter(
            model_state['densities_imag'].to(self.device)
        )
        self.gaussian_model.num_gaussians = num_gaussians
        
        # 体素化器
        use_optimized = self.config.get('use_optimized_voxelizer', False)
        if use_optimized:
            self.voxelizer = VoxelizerOptimized(
                volume_shape=self.volume_shape,
                device=self.device
            )
        else:
            self.voxelizer = Voxelizer(
                volume_shape=self.volume_shape,
                device=self.device,
                chunk_size=self.config.get('chunk_size', 2048)
            )
        
        print(f"Model restored with {num_gaussians} Gaussians")
    
    def reconstruct(self) -> torch.Tensor:
        """执行重建"""
        self.gaussian_model.eval()
        
        with torch.no_grad():
            positions, cov_inv, dens_real, dens_imag = self.gaussian_model()
            recon_image = self.voxelizer(positions, cov_inv, dens_real, dens_imag)
        
        return recon_image
    
    def evaluate(self) -> dict:
        """评估重建质量"""
        print("\n" + "="*60)
        print("Running Reconstruction...")
        print("="*60)
        
        recon_image = self.reconstruct()
        recon_abs = torch.abs(recon_image)
        
        # 计算指标
        psnr = compute_psnr(recon_abs, self.dataset.image_gt)
        ssim = compute_ssim(recon_abs, self.dataset.image_gt)
        
        # 与零填充重建比较
        psnr_zf = compute_psnr(self.dataset.image_init, self.dataset.image_gt)
        ssim_zf = compute_ssim(self.dataset.image_init, self.dataset.image_gt)
        
        # 计算 NMSE
        mse_recon = torch.mean((recon_abs - self.dataset.image_gt) ** 2)
        mse_zf = torch.mean((self.dataset.image_init - self.dataset.image_gt) ** 2)
        gt_energy = torch.mean(self.dataset.image_gt ** 2)
        
        nmse_recon = (mse_recon / gt_energy).item()
        nmse_zf = (mse_zf / gt_energy).item()
        
        results = {
            'recon_psnr': psnr,
            'recon_ssim': ssim,
            'recon_nmse': nmse_recon,
            'zerofill_psnr': psnr_zf,
            'zerofill_ssim': ssim_zf,
            'zerofill_nmse': nmse_zf,
            'psnr_improvement': psnr - psnr_zf,
            'ssim_improvement': ssim - ssim_zf,
            'acceleration': self.config.get('acceleration', 4.0),
            'num_gaussians': self.gaussian_model.num_gaussians
        }
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Acceleration: {results['acceleration']:.1f}x")
        print(f"Number of Gaussians: {results['num_gaussians']}")
        print("-"*60)
        print(f"Zero-filled: PSNR={psnr_zf:.2f} dB, SSIM={ssim_zf:.4f}, NMSE={nmse_zf:.6f}")
        print(f"3DGSMR:      PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, NMSE={nmse_recon:.6f}")
        print("-"*60)
        print(f"Improvement: PSNR=+{psnr - psnr_zf:.2f} dB, SSIM=+{ssim - ssim_zf:.4f}")
        print("="*60)
        
        return results, recon_image
    
    def save_results(self, save_dir: str, recon_image: torch.Tensor, results: dict):
        """保存重建结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存为 .mat 文件
        recon_np = recon_image.cpu().numpy()
        gt_np = self.dataset.image_gt.cpu().numpy()
        zf_np = self.dataset.image_init.cpu().numpy()
        mask_np = self.dataset.mask.cpu().numpy()
        
        mat_path = os.path.join(save_dir, 'reconstruction.mat')
        scio.savemat(mat_path, {
            'recon': recon_np,
            'gt': gt_np,
            'zerofill': zf_np,
            'mask': mask_np,
            'psnr': results['recon_psnr'],
            'ssim': results['recon_ssim'],
            'nmse': results['recon_nmse']
        })
        print(f"Results saved to: {mat_path}")
        
        # 保存为 H5 文件
        h5_path = os.path.join(save_dir, 'reconstruction.h5')
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('recon', data=recon_np)
            f.create_dataset('gt', data=gt_np)
            f.create_dataset('zerofill', data=zf_np)
            f.create_dataset('mask', data=mask_np)
            f.attrs['psnr'] = results['recon_psnr']
            f.attrs['ssim'] = results['recon_ssim']
            f.attrs['nmse'] = results['recon_nmse']
        print(f"Results saved to: {h5_path}")
        
        # 保存 JSON 报告
        import json
        json_path = os.path.join(save_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")
    
    def visualize_slices(self, recon_image: torch.Tensor, save_dir: str, num_slices: int = 5):
        """可视化切片对比"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        recon_np = torch.abs(recon_image).cpu().numpy()
        gt_np = self.dataset.image_gt.cpu().numpy()
        zf_np = self.dataset.image_init.cpu().numpy()
        
        nz, nx, ny = recon_np.shape
        
        # 选择切片位置
        slice_indices = np.linspace(nz // 4, 3 * nz // 4, num_slices, dtype=int)
        
        for i, z_idx in enumerate(slice_indices):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # GT
            axes[0].imshow(gt_np[z_idx], cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            # Zero-filled
            axes[1].imshow(zf_np[z_idx], cmap='gray')
            axes[1].set_title('Zero-filled')
            axes[1].axis('off')
            
            # 3DGSMR
            axes[2].imshow(recon_np[z_idx], cmap='gray')
            axes[2].set_title('3DGSMR')
            axes[2].axis('off')
            
            # Error map
            error = np.abs(recon_np[z_idx] - gt_np[z_idx])
            im = axes[3].imshow(error, cmap='hot')
            axes[3].set_title('Error Map')
            axes[3].axis('off')
            plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Slice {z_idx}')
            plt.tight_layout()
            
            fig_path = os.path.join(save_dir, f'slice_{z_idx:03d}.png')
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Visualization saved to: {save_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='3DGSMR Testing/Inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--num_slices', type=int, default=5,
                        help='Number of slices to visualize')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("\n" + "="*60)
    print("3DGSMR - Testing/Inference")
    print("="*60)
    
    # 创建测试器
    tester = Tester(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 评估
    results, recon_image = tester.evaluate()
    
    # 保存结果
    tester.save_results(args.save_dir, recon_image, results)
    
    # 可视化
    if args.visualize:
        vis_dir = os.path.join(args.save_dir, 'visualizations')
        tester.visualize_slices(recon_image, vis_dir, num_slices=args.num_slices)
    
    print("\nTesting completed!")


if __name__ == "__main__":
    main()
