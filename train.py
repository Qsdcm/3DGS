"""
3DGSMR Training Script
训练循环、Loss 计算、自适应控制、日志监控
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import json
from datetime import datetime

# 导入自定义模块
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
    AdaptiveController,
    compute_psnr,
    compute_ssim
)


class DataConsistencyLoss(nn.Module):
    """
    频域数据一致性损失
    
    L = || Mask · F(I_recon) - K_under ||_2^2
    
    其中:
        - Mask: 采样掩码
        - F: 3D FFT
        - I_recon: 重建图像
        - K_under: 欠采样 k-space
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        recon_image: torch.Tensor, 
        kspace_under: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算数据一致性损失
        
        Args:
            recon_image: (nz, nx, ny) 重建的复数图像
            kspace_under: (nc, nz, nx, ny) 欠采样 k-space（多线圈）
            mask: (nz, nx, ny) 采样掩码
            
        Returns:
            loss: 标量损失值
        """
        # 将重建图像扩展到多线圈（假设所有线圈图像相同，这是简化假设）
        # 实际中应该使用线圈敏感度图，这里简化处理
        nc = kspace_under.shape[0]
        
        # 对单通道图像进行 FFT
        recon_kspace = image_to_kspace(recon_image)  # (nz, nx, ny)
        
        # 扩展到多线圈
        recon_kspace_mc = recon_kspace.unsqueeze(0).expand(nc, -1, -1, -1)  # (nc, nz, nx, ny)
        
        # 应用掩码
        mask_expanded = mask.unsqueeze(0)  # (1, nz, nx, ny)
        
        # 计算损失：只在采样位置比较
        diff = mask_expanded * (recon_kspace_mc - kspace_under)
        loss = torch.mean(torch.abs(diff) ** 2)
        
        return loss


class DataConsistencyLossSingleCoil(nn.Module):
    """
    单线圈数据一致性损失（用于 SoS 合并后的数据）
    
    L = || Mask · F(I_recon) - K_combined ||_2^2
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        recon_image: torch.Tensor, 
        kspace_combined: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算数据一致性损失
        
        Args:
            recon_image: (nz, nx, ny) 重建的复数图像
            kspace_combined: (nz, nx, ny) 合并的 k-space
            mask: (nz, nx, ny) 采样掩码
            
        Returns:
            loss: 标量损失值
        """
        # FFT
        recon_kspace = image_to_kspace(recon_image)
        
        # 在采样位置计算损失
        diff = mask * (recon_kspace - kspace_combined)
        loss = torch.mean(torch.abs(diff) ** 2)
        
        return loss


class Trainer:
    """
    3DGSMR 训练器
    """
    
    def __init__(self, config: dict):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = config['device']
        
        # 创建保存目录
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 日志文件
        self.log_file = os.path.join(self.save_dir, 'training.log')
        
        # 初始化数据集
        self._init_dataset()
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 初始化损失函数
        self.loss_fn = DataConsistencyLoss()
        
        # 初始化自适应控制器
        self._init_adaptive_controller()
        
        # 训练历史
        self.history = {
            'loss': [],
            'psnr': [],
            'ssim': [],
            'num_gaussians': [],
            'split_count': [],
            'prune_count': []
        }
        
        # 记录配置
        self._log_config()
    
    def _init_dataset(self):
        """初始化数据集"""
        print("\n" + "="*60)
        print("Initializing Dataset...")
        print("="*60)
        
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
        self._print_gpu_memory("After dataset loading")
    
    def _init_model(self):
        """初始化模型"""
        print("\n" + "="*60)
        print("Initializing Model...")
        print("="*60)
        
        # 高斯模型
        init_num_gaussians = self.config.get('init_num_gaussians', 10000)
        
        self.gaussian_model = ComplexGaussianModel(
            num_gaussians=init_num_gaussians,
            volume_shape=self.volume_shape,
            device=self.device,
            init_scale=self.config.get('init_scale', 0.05),
            init_density=self.config.get('init_density', 0.01)
        )
        
        # 使用欠采样图像初始化高斯位置
        if self.config.get('use_image_init', True):
            self.gaussian_model.initialize_from_image(
                self.dataset.image_init_complex,
                threshold_ratio=self.config.get('init_threshold', 0.1),
                max_gaussians=self.config.get('max_init_gaussians', 50000),
                min_gaussians=self.config.get('min_init_gaussians', 5000)
            )
        
        # 体素化器
        use_optimized = self.config.get('use_optimized_voxelizer', False)
        if use_optimized:
            self.voxelizer = VoxelizerOptimized(
                volume_shape=self.volume_shape,
                device=self.device,
                cutoff_sigma=self.config.get('cutoff_sigma', 3.0),
                gaussian_batch_size=self.config.get('gaussian_batch_size', 256)
            )
        else:
            self.voxelizer = Voxelizer(
                volume_shape=self.volume_shape,
                device=self.device,
                chunk_size=self.config.get('chunk_size', 128),
                voxel_chunk_size=self.config.get('voxel_chunk_size', 20000),
                cutoff_sigma=self.config.get('cutoff_sigma', 3.0)
            )
        
        print(f"Gaussian Model initialized with {self.gaussian_model.num_gaussians} Gaussians")
        print(f"Voxelizer type: {'Optimized' if use_optimized else 'Standard'}")
        print(f"Voxelizer chunk_size: {self.config.get('chunk_size', 128)}, voxel_chunk_size: {self.config.get('voxel_chunk_size', 20000)}")
        
        # 显示显存使用情况
        self._print_gpu_memory("After model initialization")
    
    def _init_optimizer(self):
        """初始化优化器"""
        lr = self.config.get('learning_rate', 1e-3)
        
        # 不同参数使用不同学习率
        param_groups = [
            {'params': [self.gaussian_model.positions], 'lr': lr * 0.1, 'name': 'positions'},
            {'params': [self.gaussian_model.log_scales], 'lr': lr * 0.1, 'name': 'scales'},
            {'params': [self.gaussian_model.rotations], 'lr': lr * 0.01, 'name': 'rotations'},
            {'params': [self.gaussian_model.densities_real], 'lr': lr, 'name': 'densities_real'},
            {'params': [self.gaussian_model.densities_imag], 'lr': lr, 'name': 'densities_imag'},
        ]
        
        self.optimizer = optim.Adam(param_groups, lr=lr)
        
        # 学习率调度器
        num_iterations = self.config.get('num_iterations', 10000)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=num_iterations,
            eta_min=lr * 0.01
        )
        
        print(f"Optimizer: Adam, base LR: {lr}")
    
    def _init_adaptive_controller(self):
        """初始化自适应控制器"""
        self.adaptive_controller = AdaptiveController(
            split_grad_threshold=self.config.get('split_grad_threshold', 0.0002),
            split_scale_threshold=self.config.get('split_scale_threshold', 0.01),
            prune_density_threshold=self.config.get('prune_density_threshold', 0.001),
            prune_scale_threshold=self.config.get('prune_scale_threshold', 0.0005),
            max_gaussians=self.config.get('max_gaussians', 100000),
            min_gaussians=self.config.get('min_gaussians', 1000),
            densify_interval=self.config.get('densify_interval', 100),
            device=self.device
        )
        
        print(f"Adaptive Controller initialized")
        print(f"  - Densify interval: {self.adaptive_controller.densify_interval}")
        print(f"  - Max Gaussians: {self.adaptive_controller.max_gaussians}")
    
    def _print_gpu_memory(self, stage: str = ""):
        """打印 GPU 显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            print(f"\n[GPU Memory] {stage}")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Max Allocated: {max_allocated:.2f} GB")
            print()
    
    def _log_config(self):
        """记录配置到文件"""
        config_path = os.path.join(self.save_dir, 'config.json')
        
        # 转换不可序列化的对象
        config_to_save = {}
        for k, v in self.config.items():
            if isinstance(v, (int, float, str, bool, list, dict)):
                config_to_save[k] = v
            else:
                config_to_save[k] = str(v)
        
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"\nConfig saved to: {config_path}")
    
    def _log(self, message: str, print_msg: bool = True):
        """写入日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
        
        if print_msg:
            print(message)
    
    def train_step(self, iteration: int) -> dict:
        """
        单步训练
        
        Args:
            iteration: 当前迭代次数
            
        Returns:
            metrics: 包含损失和指标的字典
        """
        self.gaussian_model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        positions, cov_inv, dens_real, dens_imag = self.gaussian_model()
        
        # 体素化
        recon_image = self.voxelizer(positions, cov_inv, dens_real, dens_imag)
        
        # 计算损失
        loss = self.loss_fn(recon_image, self.dataset.kspace_under, self.dataset.mask)
        
        # 反向传播
        loss.backward()
        
        # 累积梯度用于自适应控制
        self.adaptive_controller.accumulate_gradients(self.gaussian_model)
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.gaussian_model.parameters(), max_norm=1.0)
        
        # 优化器步进
        self.optimizer.step()
        self.scheduler.step()
        
        # 计算指标
        with torch.no_grad():
            psnr = compute_psnr(recon_image, self.dataset.image_init_complex)
            ssim = compute_ssim(recon_image, self.dataset.image_init_complex)
            
            # 与 GT 比较
            recon_abs = torch.abs(recon_image)
            psnr_gt = compute_psnr(recon_abs, self.dataset.image_gt)
            ssim_gt = compute_ssim(recon_abs, self.dataset.image_gt)
        
        metrics = {
            'loss': loss.item(),
            'psnr': psnr_gt,
            'ssim': ssim_gt,
            'num_gaussians': self.gaussian_model.num_gaussians,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    def adaptive_step(self, iteration: int) -> dict:
        """
        执行自适应控制（分裂/剪枝）
        
        Args:
            iteration: 当前迭代次数
            
        Returns:
            stats: 分裂和剪枝统计
        """
        if not self.adaptive_controller.should_densify(iteration):
            return {'split': 0, 'prune': 0, 'total': self.gaussian_model.num_gaussians}
        
        # 执行自适应控制
        stats = self.adaptive_controller.densify_and_prune(
            self.gaussian_model, 
            self.optimizer
        )
        
        # 重新初始化优化器（因为参数数量可能变化）
        self._reinit_optimizer()
        
        return stats
    
    def _reinit_optimizer(self):
        """重新初始化优化器（参数数量变化后）"""
        lr = self.config.get('learning_rate', 1e-3)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        param_groups = [
            {'params': [self.gaussian_model.positions], 'lr': current_lr * 0.1, 'name': 'positions'},
            {'params': [self.gaussian_model.log_scales], 'lr': current_lr * 0.1, 'name': 'scales'},
            {'params': [self.gaussian_model.rotations], 'lr': current_lr * 0.01, 'name': 'rotations'},
            {'params': [self.gaussian_model.densities_real], 'lr': current_lr, 'name': 'densities_real'},
            {'params': [self.gaussian_model.densities_imag], 'lr': current_lr, 'name': 'densities_imag'},
        ]
        
        self.optimizer = optim.Adam(param_groups, lr=current_lr)
    
    def train(self):
        """主训练循环"""
        num_iterations = self.config.get('num_iterations', 10000)
        log_interval = self.config.get('log_interval', 100)
        save_interval = self.config.get('save_interval', 1000)
        
        print("\n" + "="*60)
        print("Starting Training...")
        print("="*60)
        print(f"Total iterations: {num_iterations}")
        print(f"Log interval: {log_interval}")
        print(f"Save interval: {save_interval}")
        print("="*60)
        
        # 训练前预热测试，检查显存占用
        print("\n" + "="*60)
        print("Running warmup forward pass to check GPU memory...")
        print("="*60)
        
        torch.cuda.reset_peak_memory_stats(self.device)
        self._print_gpu_memory("Before warmup")
        
        # 执行一次前向传播
        with torch.no_grad():
            positions, cov_inv, dens_real, dens_imag = self.gaussian_model()
            self._print_gpu_memory("After Gaussian forward")
            
            recon_image = self.voxelizer(positions, cov_inv, dens_real, dens_imag)
            self._print_gpu_memory("After Voxelizer forward")
            
            del recon_image
            torch.cuda.empty_cache()
        
        self._print_gpu_memory("After cleanup")
        print("Warmup completed successfully!")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        pbar = tqdm(range(1, num_iterations + 1), desc="Training")
        
        for iteration in pbar:
            # 训练步
            metrics = self.train_step(iteration)
            
            # 自适应控制
            adaptive_stats = self.adaptive_step(iteration)
            
            # 更新历史
            self.history['loss'].append(metrics['loss'])
            self.history['psnr'].append(metrics['psnr'])
            self.history['ssim'].append(metrics['ssim'])
            self.history['num_gaussians'].append(metrics['num_gaussians'])
            self.history['split_count'].append(adaptive_stats['split'])
            self.history['prune_count'].append(adaptive_stats['prune'])
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{metrics['loss']:.6f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.4f}",
                '#G': metrics['num_gaussians'],
                'Split': adaptive_stats['split'],
                'Prune': adaptive_stats['prune']
            })
            
            # 日志
            if iteration % log_interval == 0:
                elapsed = time.time() - start_time
                msg = (f"Iter {iteration}/{num_iterations} | "
                       f"Loss: {metrics['loss']:.6f} | "
                       f"PSNR: {metrics['psnr']:.2f} dB | "
                       f"SSIM: {metrics['ssim']:.4f} | "
                       f"Gaussians: {metrics['num_gaussians']} | "
                       f"Split/Prune: {adaptive_stats['split']}/{adaptive_stats['prune']} | "
                       f"LR: {metrics['lr']:.2e} | "
                       f"Time: {elapsed:.1f}s")
                self._log(msg, print_msg=False)
            
            # 保存检查点
            if iteration % save_interval == 0:
                self.save_checkpoint(iteration)
        
        # 最终保存
        self.save_checkpoint(num_iterations, final=True)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Final PSNR: {self.history['psnr'][-1]:.2f} dB")
        print(f"Final SSIM: {self.history['ssim'][-1]:.4f}")
        print(f"Final Gaussians: {self.history['num_gaussians'][-1]}")
        print(f"{'='*60}")
    
    def save_checkpoint(self, iteration: int, final: bool = False):
        """保存检查点"""
        suffix = 'final' if final else f'iter_{iteration}'
        ckpt_path = os.path.join(self.save_dir, f'checkpoint_{suffix}.pth')
        
        checkpoint = {
            'iteration': iteration,
            'model_state': {
                'positions': self.gaussian_model.positions.data,
                'log_scales': self.gaussian_model.log_scales.data,
                'rotations': self.gaussian_model.rotations.data,
                'densities_real': self.gaussian_model.densities_real.data,
                'densities_imag': self.gaussian_model.densities_imag.data,
                'num_gaussians': self.gaussian_model.num_gaussians
            },
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, ckpt_path)
        self._log(f"Checkpoint saved: {ckpt_path}", print_msg=True)
    
    def load_checkpoint(self, ckpt_path: str):
        """加载检查点"""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # 恢复模型
        model_state = checkpoint['model_state']
        self.gaussian_model.num_gaussians = model_state['num_gaussians']
        self.gaussian_model.positions = nn.Parameter(model_state['positions'])
        self.gaussian_model.log_scales = nn.Parameter(model_state['log_scales'])
        self.gaussian_model.rotations = nn.Parameter(model_state['rotations'])
        self.gaussian_model.densities_real = nn.Parameter(model_state['densities_real'])
        self.gaussian_model.densities_imag = nn.Parameter(model_state['densities_imag'])
        
        # 恢复历史
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from: {ckpt_path}")
        print(f"Resumed at iteration: {checkpoint['iteration']}")
        
        return checkpoint['iteration']
    
    def reconstruct(self) -> torch.Tensor:
        """执行重建"""
        self.gaussian_model.eval()
        
        with torch.no_grad():
            positions, cov_inv, dens_real, dens_imag = self.gaussian_model()
            recon_image = self.voxelizer(positions, cov_inv, dens_real, dens_imag)
        
        return recon_image
    
    def evaluate(self) -> dict:
        """评估重建质量"""
        recon_image = self.reconstruct()
        
        recon_abs = torch.abs(recon_image)
        
        psnr = compute_psnr(recon_abs, self.dataset.image_gt)
        ssim = compute_ssim(recon_abs, self.dataset.image_gt)
        
        # 与零填充重建比较
        psnr_zf = compute_psnr(self.dataset.image_init, self.dataset.image_gt)
        ssim_zf = compute_ssim(self.dataset.image_init, self.dataset.image_gt)
        
        results = {
            'recon_psnr': psnr,
            'recon_ssim': ssim,
            'zerofill_psnr': psnr_zf,
            'zerofill_ssim': ssim_zf,
            'psnr_improvement': psnr - psnr_zf,
            'ssim_improvement': ssim - ssim_zf
        }
        
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Zero-filled: PSNR={psnr_zf:.2f} dB, SSIM={ssim_zf:.4f}")
        print(f"3DGSMR:      PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
        print(f"Improvement: PSNR=+{psnr - psnr_zf:.2f} dB, SSIM=+{ssim - ssim_zf:.4f}")
        print("="*60)
        
        return results


def get_default_config() -> dict:
    """获取默认配置"""
    return {
        # 数据
        'data_path': '/data/data54/wanghaobo/data/ksp_full.h5',
        'acceleration': 4.0,
        'center_fraction': 0.08,
        'mask_type': 'gaussian',
        'seed': 42,
        
        # 模型
        'init_num_gaussians': 10000,
        'init_scale': 0.05,
        'init_density': 0.01,
        'use_image_init': True,
        'init_threshold': 0.1,
        'max_init_gaussians': 50000,
        'min_init_gaussians': 5000,
        
        # 体素化器 - 分块大小设置以控制显存占用
        'use_optimized_voxelizer': False,
        'chunk_size': 512,           # 高斯分块大小 (可分离高斯近似后可使用较大值)
        'voxel_chunk_size': 20000,   # 未使用，保留以兼容旧接口
        'cutoff_sigma': 3.0,
        
        # 优化器
        'learning_rate': 1e-3,
        'num_iterations': 5000,
        
        # 自适应控制
        'split_grad_threshold': 0.0002,
        'split_scale_threshold': 0.01,
        'prune_density_threshold': 0.001,
        'prune_scale_threshold': 0.0005,
        'max_gaussians': 100000,
        'min_gaussians': 1000,
        'densify_interval': 100,
        
        # 日志
        'log_interval': 100,
        'save_interval': 1000,
        'save_dir': 'checkpoints',
        
        # 设备
        'device': 'cuda:1'
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='3DGSMR Training')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                        default='/data/data54/wanghaobo/data/ksp_full.h5',
                        help='Path to k-space H5 file')
    parser.add_argument('--acceleration', type=float, default=4.0,
                        help='Undersampling acceleration factor')
    parser.add_argument('--center_fraction', type=float, default=0.08,
                        help='Fraction of center k-space to fully sample')
    parser.add_argument('--mask_type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform'],
                        help='Type of undersampling mask')
    
    # 模型参数
    parser.add_argument('--init_num_gaussians', type=int, default=10000,
                        help='Initial number of Gaussians')
    parser.add_argument('--max_gaussians', type=int, default=100000,
                        help='Maximum number of Gaussians')
    
    # 训练参数
    parser.add_argument('--num_iterations', type=int, default=5000,
                        help='Number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--densify_interval', type=int, default=100,
                        help='Interval for adaptive densification')
    
    # 日志参数
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Checkpoint save interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 构建配置
    config = get_default_config()
    
    # 用命令行参数覆盖默认配置
    for key, value in vars(args).items():
        config[key] = value
    
    # 打印配置
    print("\n" + "="*60)
    print("3DGSMR - 3D Gaussian Splatting for MRI Reconstruction")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # 创建训练器并训练
    trainer = Trainer(config)
    trainer.train()
    
    # 评估
    results = trainer.evaluate()
    
    # 保存评估结果
    results_path = os.path.join(config['save_dir'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
