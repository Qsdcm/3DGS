"""
3DGSMR Testing/Inference Entry Point

用法:
    python test.py --checkpoint checkpoints/best.pth
    python test.py --checkpoint checkpoints/best.pth --output_dir results/

功能:
1. 加载训练好的模型
2. 执行重建
3. 计算评估指标 (PSNR, SSIM, NMSE)
4. 保存重建结果
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any

from data import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian import GaussianModel3D, Voxelizer
from metrics import evaluate_reconstruction, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test 3DGSMR for MRI Reconstruction'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (if not stored in checkpoint)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Override data path'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./test_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    parser.add_argument(
        '--save_volume',
        action='store_true',
        default=True,
        help='Save reconstructed volume as .npy'
    )
    
    parser.add_argument(
        '--save_slices',
        action='store_true',
        default=False,
        help='Save slice images as PNG'
    )
    
    return parser.parse_args()


class GaussianTester:
    """
    3DGSMR测试/推理类
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Dict[str, Any] = None,
        device: torch.device = None,
        data_path: str = None
    ):
        """
        Args:
            checkpoint_path: Checkpoint文件路径
            config: 配置字典 (如果为None，从checkpoint加载)
            device: 计算设备
            data_path: 数据路径 (覆盖配置)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 加载checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        if config is not None:
            self.config = config
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found. Please provide config file.")
        
        # 覆盖数据路径
        if data_path is not None:
            self.config['data']['data_path'] = data_path
        
        # 初始化组件
        self._setup_data()
        self._setup_model(checkpoint)
        
        print(f"Loaded model with {self.gaussian_model.num_points} Gaussians")
        print(f"Checkpoint iteration: {checkpoint.get('iteration', 'unknown')}")
        
    def _setup_data(self):
        """初始化数据"""
        data_config = self.config['data']
        
        # 加载数据集
        self.dataset = MRIDataset(
            data_path=data_config['data_path'],
            acceleration_factor=data_config['acceleration_factor'],
            mask_type=data_config.get('mask_type', 'gaussian'),
            use_acs=data_config.get('use_acs', True),
            acs_lines=int(data_config.get('center_fraction', 0.08) * 256)
        )
        
        # 获取数据
        data = self.dataset.get_data()
        
        self.kspace_full = data['kspace_full'].to(self.device)
        self.kspace_undersampled = data['kspace_undersampled'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.volume_shape = data['volume_shape']
        
        # 目标图像
        self.target_image = data['ground_truth'].to(self.device)
        
        # 零填充重建 (baseline)
        self.zero_filled = data['zero_filled'].to(self.device)
        
    def _setup_model(self, checkpoint: Dict):
        """初始化模型"""
        gaussian_config = self.config['gaussian']
        
        # 从checkpoint获取模型信息
        gaussian_state = checkpoint['gaussian_state']
        num_points = gaussian_state['positions'].shape[0]
        
        # 创建Gaussian模型
        self.gaussian_model = GaussianModel3D(
            num_points=num_points,
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        
        # 加载状态
        self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        
        # 创建Voxelizer
        self.voxelizer = Voxelizer(
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        
    def reconstruct(self) -> torch.Tensor:
        """
        执行重建
        
        Returns:
            重建的体积数据
        """
        self.gaussian_model.eval()
        
        with torch.no_grad():
            # 获取参数
            positions = self.gaussian_model.positions
            scales = self.gaussian_model.get_scales()
            rotations = self.gaussian_model.rotations
            density = self.gaussian_model.get_densities()
            
            # 渲染Gaussians到体素
            volume = self.voxelizer(
                positions=positions,
                scales=scales,
                rotations=rotations,
                density=density
            )
        
        return volume
    
    def evaluate(self, volume: torch.Tensor = None) -> Dict[str, float]:
        """
        评估重建质量
        
        Args:
            volume: 重建体积 (如果为None则执行重建)
            
        Returns:
            评估指标字典
        """
        if volume is None:
            volume = self.reconstruct()
        
        # 计算指标
        metrics = evaluate_reconstruction(
            pred=volume,
            target=self.target_image,
            compute_3d_ssim=True
        )
        
        return metrics
    
    def evaluate_zero_filled(self) -> Dict[str, float]:
        """评估零填充重建 (baseline)"""
        metrics = evaluate_reconstruction(
            pred=self.zero_filled,
            target=self.target_image,
            compute_3d_ssim=True
        )
        return metrics
    
    def save_results(
        self,
        output_dir: str,
        save_volume: bool = True,
        save_slices: bool = False
    ):
        """
        保存重建结果
        
        Args:
            output_dir: 输出目录
            save_volume: 是否保存体积数据
            save_slices: 是否保存切片图像
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 重建
        volume = self.reconstruct()
        
        # 评估
        metrics = self.evaluate(volume)
        zf_metrics = self.evaluate_zero_filled()
        
        # 打印结果
        print("\n" + "=" * 50)
        print("Reconstruction Results")
        print("=" * 50)
        print("\n3DGSMR Reconstruction:")
        print_metrics(metrics, prefix="  ")
        print("\nZero-filled Baseline:")
        print_metrics(zf_metrics, prefix="  ")
        print(f"\nPSNR Improvement: {metrics['psnr'] - zf_metrics['psnr']:.2f} dB")
        print("=" * 50)
        
        # 保存评估结果
        results = {
            '3dgsmr': metrics,
            'zero_filled': zf_metrics,
            'improvement': {
                'psnr': metrics['psnr'] - zf_metrics['psnr'],
                'ssim': metrics['ssim'] - zf_metrics['ssim']
            },
            'num_gaussians': self.gaussian_model.num_points
        }
        
        results_path = os.path.join(output_dir, 'metrics.yaml')
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Metrics saved to: {results_path}")
        
        # 保存体积数据
        if save_volume:
            volume_np = volume.detach().cpu().numpy()
            
            # 保存complex volume
            volume_path = os.path.join(output_dir, 'reconstruction.npy')
            np.save(volume_path, volume_np)
            print(f"Volume saved to: {volume_path}")
            
            # 保存幅度图
            magnitude_path = os.path.join(output_dir, 'reconstruction_magnitude.npy')
            np.save(magnitude_path, np.abs(volume_np))
            print(f"Magnitude saved to: {magnitude_path}")
            
            # 保存目标和零填充用于比较
            target_np = self.target_image.detach().cpu().numpy()
            np.save(os.path.join(output_dir, 'target.npy'), target_np)
            
            zf_np = self.zero_filled.detach().cpu().numpy()
            np.save(os.path.join(output_dir, 'zero_filled.npy'), zf_np)
        
        # 保存切片图像
        if save_slices:
            self._save_slice_images(volume, output_dir)
        
        return metrics
    
    def _save_slice_images(self, volume: torch.Tensor, output_dir: str):
        """保存切片图像"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available. Skipping slice images.")
            return
        
        slice_dir = os.path.join(output_dir, 'slices')
        os.makedirs(slice_dir, exist_ok=True)
        
        # 转换
        volume_np = torch.abs(volume).detach().cpu().numpy()
        target_np = torch.abs(self.target_image).detach().cpu().numpy()
        zf_np = torch.abs(self.zero_filled).detach().cpu().numpy()
        
        # 选择中间切片
        d, h, w = volume_np.shape
        mid_slices = [d // 4, d // 2, 3 * d // 4]
        
        for idx, slice_idx in enumerate(mid_slices):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(target_np[slice_idx], cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            axes[1].imshow(zf_np[slice_idx], cmap='gray')
            axes[1].set_title('Zero-filled')
            axes[1].axis('off')
            
            axes[2].imshow(volume_np[slice_idx], cmap='gray')
            axes[2].set_title('3DGSMR')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(slice_dir, f'slice_{slice_idx:03d}.png'),
                dpi=150,
                bbox_inches='tight'
            )
            plt.close()
        
        print(f"Slice images saved to: {slice_dir}")


def main():
    # 解析参数
    args = parse_args()
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 加载配置
    config = None
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # 创建Tester
    tester = GaussianTester(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device,
        data_path=args.data_path
    )
    
    # 保存结果
    metrics = tester.save_results(
        output_dir=args.output_dir,
        save_volume=args.save_volume,
        save_slices=args.save_slices
    )
    
    print("\nTesting complete!")


if __name__ == '__main__':
    main()
