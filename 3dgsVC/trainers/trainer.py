"""
3DGSMR Trainer (Final Revised Version)

完全对齐论文:
1. Loss使用 Sum reduction (配合losses.py修改) -> 梯度量级正常
2. 传递极小的 scale_threshold (0.0005) -> 允许细微结构分裂
3. Densification频率=100iter, 持续到2500iter -> 确保长到400k点
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import numpy as np
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm

from data import MRIDataset
from data.transforms import fft3c, ifft3c
from gaussian import GaussianModel3D, Voxelizer
from losses import CombinedLoss
from metrics import evaluate_reconstruction

class GaussianTrainer:
    """
    3D Gaussian MRI重建训练器
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"Using device: {self.device}")
        
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()
        
        self.current_iteration = 0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        
    def _setup_data(self):
        data_config = self.config['data']
        self.dataset = MRIDataset(
            data_path=data_config['data_path'],
            acceleration_factor=data_config['acceleration_factor'],
            mask_type=data_config.get('mask_type', 'gaussian'),
            use_acs=data_config.get('use_acs', True),
            acs_lines=int(data_config.get('center_fraction', 0.08) * 256)
        )
        data = self.dataset.get_data()
        
        self.kspace_full = data['kspace_full'].to(self.device)
        self.kspace_undersampled = data['kspace_undersampled'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.volume_shape = data['volume_shape']
        self.target_image = data['ground_truth'].to(self.device)
        self.zero_filled = data['zero_filled'].to(self.device)
        
        print(f"Volume shape: {self.volume_shape}")
        print(f"K-space shape: {self.kspace_full.shape}")
        print(f"Acceleration factor: {data_config['acceleration_factor']}")
        
    def _setup_model(self):
        gaussian_config = self.config['gaussian']
        init_method = gaussian_config.get('init_method', 'from_image')
        
        if init_method == 'from_image':
            self.gaussian_model = GaussianModel3D.from_image(
                image=self.zero_filled,
                num_points=gaussian_config['initial_num_points'],
                initial_scale=gaussian_config.get('initial_scale', 2.0),
                device=str(self.device)
            )
        else:
            self.gaussian_model = GaussianModel3D(
                num_points=gaussian_config['initial_num_points'],
                volume_shape=tuple(self.volume_shape),
                initial_scale=gaussian_config.get('initial_scale', 2.0),
                device=str(self.device)
            )
        
        self.voxelizer = Voxelizer(
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        print(f"Initialized with {self.gaussian_model.num_points} Gaussians")
        
    def _setup_loss(self):
        loss_config = self.config['loss']
        self.criterion = CombinedLoss(
            kspace_weight=loss_config.get('kspace_weight', 1.0),
            image_weight=loss_config.get('image_weight', 0.0),
            tv_weight=loss_config.get('tv_weight', 0.0),
            loss_type=loss_config.get('loss_type', 'l2') # 论文倾向 L2
        ).to(self.device)
        
    def _setup_optimizer(self):
        train_config = self.config['training']
        gaussian_config = self.config['gaussian']
        params = self.gaussian_model.get_optimizable_params(
            lr_position=gaussian_config.get('position_lr', 1e-4),
            lr_density=gaussian_config.get('density_lr', 1e-3),
            lr_scale=gaussian_config.get('scale_lr', 5e-4),
            lr_rotation=gaussian_config.get('rotation_lr', 1e-4)
        )
        self.optimizer = optim.Adam(params)
        
        scheduler_config = train_config.get('lr_scheduler', {})
        scheduler_type = scheduler_config.get('type', 'exponential')
        if scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=scheduler_config.get('gamma', 0.999)
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['max_iterations']
            )
            
    def _setup_output(self):
        output_config = self.config['output']
        self.output_dir = output_config['output_dir']
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.result_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        config_path = os.path.join(self.output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = self.gaussian_model.positions
        if hasattr(self.gaussian_model, 'get_scales'):
            scales = self.gaussian_model.get_scales()
        else:
            scales = self.gaussian_model.get_scale_values()
            
        rotations = self.gaussian_model.rotations
        if hasattr(self.gaussian_model, 'get_densities'):
            density = self.gaussian_model.get_densities()
        else:
            density = self.gaussian_model.density
        
        volume = self.voxelizer(
            positions=positions,
            scales=scales,
            rotations=rotations,
            density=density
        )
        kspace = fft3c(volume)
        return volume, kspace
    
    def compute_gradient_stats(self) -> Dict[str, torch.Tensor]:
        """计算梯度统计信息，增加 mean_grad 用于调试"""
        if self.gaussian_model.positions.grad is None:
            return {}
        
        grad_norm = torch.norm(self.gaussian_model.positions.grad, dim=-1)
        return {
            'grad_norm': grad_norm,
            'mean_grad': grad_norm.mean(), # 关键调试指标
            'max_grad': grad_norm.max()
        }
    
    def adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        """
        自适应密度控制 (Paper Section IV)
        """
        adaptive_config = self.config['adaptive_control']
        if not adaptive_config.get('enable', True):
            return {'split': 0, 'clone': 0, 'prune': 0}

        stats = {'split': 0, 'clone': 0, 'prune': 0}
        
        # 参数读取
        start_iter = adaptive_config.get('densify_from_iter', 100)
        end_iter = adaptive_config.get('densify_until_iter', 2500) # 延长到2500
        interval = adaptive_config.get('densify_every', 100)
        max_num_points = self.config['gaussian'].get('max_num_points', 400000)

        # 硬性上限检查
        if self.gaussian_model.num_points >= max_num_points:
            return stats
        
        if iteration < start_iter or iteration > end_iter:
            return stats
        
        if iteration % interval != 0:
            return stats
        
        grad_stats = self.compute_gradient_stats()
        if not grad_stats:
            return stats
        
        grad_norm = grad_stats['grad_norm']
        # 梯度阈值: 如果Loss是Sum, 梯度会很大, 0.0002很容易满足
        grad_threshold = adaptive_config.get('grad_threshold', 0.0002)
        
        if hasattr(self.gaussian_model, 'get_scale_values'):
            scales = self.gaussian_model.get_scale_values()
        else:
            scales = self.gaussian_model.get_scales()
        max_scale = scales.max(dim=-1)[0]
        
        # 关键: 读取极小的 scale_threshold (例如 0.0005)
        # 必须确保 config 中设置了这个值，否则分裂会被默认值 0.01 拦截
        scale_threshold = adaptive_config.get('scale_threshold', 0.0005)
        max_scale_limit = adaptive_config.get('max_scale', 0.5)
        
        high_grad_mask = grad_norm > grad_threshold
        
        # --- Split (Long-axis) ---
        if adaptive_config.get('use_long_axis_splitting', True):
            if high_grad_mask.shape[0] == self.gaussian_model.num_points:
                # 只有梯度大且尺度也够大的点才分裂
                # 如果 scale_threshold 太大 (如 0.01)，小点就永远无法分裂了
                split_mask = high_grad_mask & (max_scale > scale_threshold)
                
                if split_mask.sum() > 0:
                    # 预测分裂后的数量
                    if self.gaussian_model.num_points + split_mask.sum() <= max_num_points:
                        self.gaussian_model.densify_and_split(
                            grads=grad_norm,
                            grad_threshold=grad_threshold,
                            scale_threshold=scale_threshold, # 传递更小的阈值
                            use_long_axis_splitting=True
                        )
                        stats['split'] = split_mask.sum().item()
                        high_grad_mask = None # 消耗掉mask
        
        # --- Clone (High Accel usually disabled) ---
        if adaptive_config.get('use_cloning', False) and high_grad_mask is not None:
             if hasattr(self.gaussian_model, 'get_scale_values'):
                scales = self.gaussian_model.get_scale_values()
             else:
                scales = self.gaussian_model.get_scales()
             max_scale = scales.max(dim=-1)[0]
            
             if high_grad_mask.shape[0] == self.gaussian_model.num_points:
                clone_mask = high_grad_mask & (max_scale <= scale_threshold)
                if clone_mask.sum() > 0:
                    if self.gaussian_model.num_points + clone_mask.sum() <= max_num_points:
                        self.gaussian_model.densify_and_clone(grad_norm, grad_threshold, scale_threshold)
                        stats['clone'] = clone_mask.sum().item()
        
        # --- Prune ---
        opacity_threshold = adaptive_config.get('opacity_threshold', 0.01)
        
        if hasattr(self.gaussian_model, 'get_densities'):
            densities = torch.abs(self.gaussian_model.get_densities())
        else:
            densities = torch.abs(self.gaussian_model.density)
            
        if hasattr(self.gaussian_model, 'get_scale_values'):
            scales = self.gaussian_model.get_scale_values()
        else:
            scales = self.gaussian_model.get_scales()
        max_scale = scales.max(dim=-1)[0]
        
        prune_mask = (densities < opacity_threshold) | (max_scale > max_scale_limit)
        
        if (self.gaussian_model.num_points - prune_mask.sum()) >= 100:
            if prune_mask.sum() > 0:
                self.gaussian_model.prune(opacity_threshold)
                stats['prune'] = prune_mask.sum().item()
        
        # 只要结构变了，就必须重建优化器
        if stats['split'] > 0 or stats['clone'] > 0 or stats['prune'] > 0:
            train_config = self.config['training']
            gaussian_config = self.config['gaussian']
            params = self.gaussian_model.get_optimizable_params(
                lr_position=gaussian_config.get('position_lr', 1e-4),
                lr_density=gaussian_config.get('density_lr', 1e-3),
                lr_scale=gaussian_config.get('scale_lr', 5e-4),
                lr_rotation=gaussian_config.get('rotation_lr', 1e-4)
            )
            self.optimizer = optim.Adam(params)
        
        return stats
    
    def train_step(self) -> Dict[str, float]:
        self.gaussian_model.train()
        self.optimizer.zero_grad()
        
        volume, kspace_pred = self.forward()
        
        loss_dict = self.criterion(
            kspace_pred=kspace_pred,
            kspace_target=self.kspace_undersampled,
            mask=self.mask,
            image_pred=volume,
            image_target=self.target_image
        )
        
        loss_dict['total_loss'].backward()
        
        # 梯度裁剪
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.gaussian_model.parameters(),
                max_grad_norm
            )
            
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}
    
    def evaluate(self) -> Dict[str, float]:
        self.gaussian_model.eval()
        with torch.no_grad():
            volume, kspace_pred = self.forward()
            metrics = evaluate_reconstruction(
                pred=volume,
                target=self.target_image,
                compute_3d_ssim=True
            )
        return metrics
    
    def save_checkpoint(self, iteration: int, is_best: bool = False):
        checkpoint = {
            'iteration': iteration,
            'gaussian_state': self.gaussian_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
        
        save_every = self.config['training'].get('save_every', 500)
        if iteration % save_every == 0:
            iter_path = os.path.join(self.checkpoint_dir, f'checkpoint_{iteration:06d}.pth')
            torch.save(checkpoint, iter_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.current_iteration = checkpoint['iteration']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.best_ssim = checkpoint.get('best_ssim', 0.0)
        print(f"Loaded checkpoint from iteration {self.current_iteration}")
    
    def save_reconstruction(self, iteration: int):
        self.gaussian_model.eval()
        with torch.no_grad():
            volume, kspace_pred = self.forward()
            volume_np = volume.detach().cpu().numpy()
            result_path = os.path.join(self.result_dir, f'reconstruction_{iteration:06d}.npy')
            np.save(result_path, volume_np)
            final_path = os.path.join(self.result_dir, 'reconstruction_final.npy')
            np.save(final_path, volume_np)
    
    def train(self, resume_from: Optional[str] = None):
        """完整训练流程"""
        train_config = self.config['training']
        max_iterations = train_config['max_iterations']
        eval_every = train_config.get('eval_every', 100)
        log_every = train_config.get('log_every', 50)
        
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        start_iter = self.current_iteration
        print(f"\nStarting training from iteration {start_iter}")
        print(f"Total iterations: {max_iterations}")
        print("-" * 50)
        
        pbar = tqdm(range(start_iter, max_iterations), desc="Training", dynamic_ncols=True)
        
        for iteration in pbar:
            self.current_iteration = iteration
            
            loss_dict = self.train_step()
            adaptive_stats = self.adaptive_density_control(iteration)
            self.scheduler.step()
            
            # 增强的Log信息
            if iteration % log_every == 0:
                grad_stats = self.compute_gradient_stats()
                mean_grad = grad_stats.get('mean_grad', 0.0)
                if isinstance(mean_grad, torch.Tensor):
                    mean_grad = mean_grad.item()
                    
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.2e}", # 科学计数法看大数
                    'grad': f"{mean_grad:.2e}",             # 监控梯度是否 > 0.0002
                    'n_pts': self.gaussian_model.num_points
                })
            
            if iteration % eval_every == 0 or iteration == max_iterations - 1:
                metrics = self.evaluate()
                is_best = metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = metrics['psnr']
                    self.best_ssim = metrics['ssim']
                
                print(f"\n[Iter {iteration}] PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
                
                # 打印分裂详情
                if adaptive_stats['split'] > 0 or adaptive_stats['clone'] > 0 or adaptive_stats['prune'] > 0:
                    print(f"  Density: split={adaptive_stats['split']}, clone={adaptive_stats['clone']}, prune={adaptive_stats['prune']}")
                    
                print(f"  Num Gaussians: {self.gaussian_model.num_points}")
                
                self.save_checkpoint(iteration, is_best)
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        self.save_reconstruction(max_iterations)
        return self.best_psnr, self.best_ssim