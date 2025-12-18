"""
3DGSMR Trainer

完整的训练流程:
1. 初始化3D Gaussians (从零填充重建或随机)
2. 迭代优化:
   - 渲染Gaussians到体素
   - FFT得到k-space
   - 计算k-space loss
   - 反向传播更新参数
   - 自适应密度控制 (split/clone/prune)
3. 保存checkpoints和重建结果
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
from metrics import evaluate_reconstruction, print_metrics


class GaussianTrainer:
    """
    3D Gaussian MRI重建训练器
    
    实现论文中的完整训练流程:
    - 前向渲染: Gaussians -> Volume -> K-space
    - K-space域loss
    - 自适应密度控制 (论文Section IV)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        """
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print(f"Using device: {self.device}")
        
        # 初始化组件
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        self._setup_output()
        
        # 训练状态
        self.current_iteration = 0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        
    def _setup_data(self):
        """初始化数据"""
        data_config = self.config['data']
        
        # 加载数据集
        self.dataset = MRIDataset(
            data_path=data_config['data_path'],
            acceleration_factor=data_config['acceleration_factor'],
            mask_type=data_config.get('mask_type', 'gaussian'),
            use_acs=data_config.get('use_acs', True),
            acs_lines=int(data_config.get('center_fraction', 0.08) * 256)  # 转换center_fraction为acs_lines
        )
        
        # 获取数据
        data = self.dataset.get_data()
        
        self.kspace_full = data['kspace_full'].to(self.device)
        self.kspace_undersampled = data['kspace_undersampled'].to(self.device)
        self.mask = data['mask'].to(self.device)
        self.volume_shape = data['volume_shape']
        
        # 目标图像 (用于评估)
        self.target_image = data['ground_truth'].to(self.device)
        
        # 零填充重建 (用于初始化)
        self.zero_filled = data['zero_filled'].to(self.device)
        
        print(f"Volume shape: {self.volume_shape}")
        print(f"K-space shape: {self.kspace_full.shape}")
        print(f"Acceleration factor: {data_config['acceleration_factor']}")
        
    def _setup_model(self):
        """初始化Gaussian模型"""
        gaussian_config = self.config['gaussian']
        
        # 从零填充重建初始化
        init_method = gaussian_config.get('init_method', 'from_image')
        
        if init_method == 'from_image':
            # 使用类方法从图像初始化
            self.gaussian_model = GaussianModel3D.from_image(
                image=self.zero_filled,
                num_points=gaussian_config['initial_num_points'],
                initial_scale=gaussian_config.get('init_scale', 2.0),
                device=str(self.device)
            )
        else:  # random
            # 随机初始化
            self.gaussian_model = GaussianModel3D(
                num_points=gaussian_config['initial_num_points'],
                volume_shape=tuple(self.volume_shape),
                initial_scale=gaussian_config.get('init_scale', 2.0),
                device=str(self.device)
            )
        
        # 创建Voxelizer
        self.voxelizer = Voxelizer(
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        
        print(f"Initialized with {self.gaussian_model.num_points} Gaussians")
        
    def _setup_loss(self):
        """初始化损失函数"""
        loss_config = self.config['loss']
        
        self.criterion = CombinedLoss(
            kspace_weight=loss_config.get('kspace_weight', 1.0),
            image_weight=loss_config.get('image_weight', 0.0),
            tv_weight=loss_config.get('tv_weight', 0.0),
            loss_type=loss_config.get('loss_type', 'l1')
        ).to(self.device)
        
    def _setup_optimizer(self):
        """初始化优化器"""
        train_config = self.config['training']
        
        # 获取可优化参数
        params = self.gaussian_model.get_optimizable_params(
            lr_position=train_config.get('lr_position', 1e-4),
            lr_density=train_config.get('lr_density', 1e-3),
            lr_scale=train_config.get('lr_scale', 5e-4),
            lr_rotation=train_config.get('lr_rotation', 1e-4)
        )
        
        self.optimizer = optim.Adam(params)
        
        # 学习率调度器
        scheduler_type = train_config.get('scheduler', 'exponential')
        if scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=train_config.get('lr_decay', 0.999)
            )
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['max_iterations']
            )
            
    def _setup_output(self):
        """设置输出目录"""
        output_config = self.config['output']
        
        self.output_dir = output_config['output_dir']
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.result_dir = os.path.join(self.output_dir, 'results')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(self.output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: Gaussians -> Volume -> K-space
        
        论文公式(3): x_j = sum_i G_i^3(j|ρ_i, p_i, Σ_i)
        
        Returns:
            (reconstructed_volume, predicted_kspace)
        """
        # 获取Gaussian参数
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
        
        # FFT到k-space
        kspace = fft3c(volume)
        
        return volume, kspace
    
    def compute_gradient_stats(self) -> Dict[str, torch.Tensor]:
        """计算梯度统计信息用于自适应密度控制"""
        if self.gaussian_model.positions.grad is None:
            return {}
        
        # 位置梯度的范数
        grad_norm = torch.norm(
            self.gaussian_model.positions.grad, dim=-1
        )
        
        return {
            'grad_norm': grad_norm,
            'mean_grad': grad_norm.mean(),
            'max_grad': grad_norm.max()
        }
    
    def adaptive_density_control(self, iteration: int) -> Dict[str, int]:
        """
        自适应密度控制
        
        论文Section IV的核心策略:
        - Split: 将高梯度、大尺度的Gaussian沿长轴分裂
        - Clone: 复制高梯度、小尺度的Gaussian
        - Prune: 移除低密度或过大的Gaussian
        
        Returns:
            操作统计信息
        """
        adaptive_config = self.config['adaptive_control']
        stats = {'split': 0, 'clone': 0, 'prune': 0}
        
        # 检查是否在控制区间内
        start_iter = adaptive_config.get('densify_start', 100)
        end_iter = adaptive_config.get('densify_end', 1500)
        interval = adaptive_config.get('densify_interval', 100)
        
        if iteration < start_iter or iteration > end_iter:
            return stats
        
        if iteration % interval != 0:
            return stats
        
        # 获取梯度统计
        grad_stats = self.compute_gradient_stats()
        if not grad_stats:
            return stats
        
        grad_norm = grad_stats['grad_norm']
        grad_threshold = adaptive_config.get('grad_threshold', 0.0002)
        
        # 获取当前尺度
        scales = self.gaussian_model.get_scale_values()  # 使用get_scale_values获取真实尺度
        max_scale = scales.max(dim=-1)[0]
        
        # 尺度阈值
        scale_threshold = adaptive_config.get('scale_threshold', 0.01)
        max_scale_limit = adaptive_config.get('max_scale', 0.1)
        
        # 高梯度点mask
        high_grad_mask = grad_norm > grad_threshold
        
        # Split: 高梯度且尺度大的Gaussian
        # 论文特别强调沿长轴分裂对高加速因子有效
        if adaptive_config.get('long_axis_splitting', True):
            # 重新计算mask，因为点数可能变了
            # 注意：这里简化处理，如果刚发生了split，梯度信息可能不再对应
            # 理想情况下应该重新计算梯度，或者只执行一种操作
            # 这里我们优先执行split，如果执行了split，就不执行clone
            
            # 重新获取当前状态
            scales = self.gaussian_model.get_scale_values()
            max_scale = scales.max(dim=-1)[0]
            
            # 确保mask长度匹配
            if high_grad_mask.shape[0] != self.gaussian_model.num_points:
                # 如果点数变了（虽然这里不应该变），则跳过
                pass
            else:
                split_mask = high_grad_mask & (max_scale > scale_threshold)
                if split_mask.sum() > 0:
                    self.gaussian_model.densify_and_split(split_mask)
                    stats['split'] = split_mask.sum().item()
                    
                    # 如果发生了split，点数增加了，原来的mask失效了
                    # 为了简单起见，本次迭代不再执行clone
                    high_grad_mask = None 
        
        # Clone: 高梯度且尺度小的Gaussian
        if adaptive_config.get('enable_cloning', True) and high_grad_mask is not None:
            # 重新获取当前状态
            scales = self.gaussian_model.get_scale_values()
            max_scale = scales.max(dim=-1)[0]
            
            if high_grad_mask.shape[0] == self.gaussian_model.num_points:
                clone_mask = high_grad_mask & (max_scale <= scale_threshold)
                if clone_mask.sum() > 0:
                    self.gaussian_model.densify_and_clone(clone_mask)
                    stats['clone'] = clone_mask.sum().item()
        
        # Prune: 移除低密度或过大的Gaussian
        prune_threshold = adaptive_config.get('prune_threshold', 0.005)
        densities = torch.abs(self.gaussian_model.get_densities())
        
        # 重新获取scales，因为可能发生了变化
        scales = self.gaussian_model.get_scale_values()
        max_scale = scales.max(dim=-1)[0]
        
        prune_mask = (densities < prune_threshold) | (max_scale > max_scale_limit)
        
        # 保持最小数量的Gaussians
        min_points = adaptive_config.get('min_num_points', 100)
        if (self.gaussian_model.num_points - prune_mask.sum()) >= min_points:
            if prune_mask.sum() > 0:
                self.gaussian_model.prune(prune_mask)
                stats['prune'] = prune_mask.sum().item()
        
        # 重建优化器
        if stats['split'] > 0 or stats['clone'] > 0 or stats['prune'] > 0:
            # 关键修复：当参数形状改变时，必须清除优化器状态
            # 并且不能在backward之后立即改变参数形状，因为梯度还在计算图中
            # 但这里是在backward之后调用的（在train loop中），所以是安全的
            # 问题在于：如果我们在backward之前改变了参数（比如在adaptive_density_control中），
            # 那么backward时就会报错，因为计算图中的参数形状和实际参数形状不匹配
            
            # 实际上，adaptive_density_control是在train loop的末尾调用的
            # 所以参数改变发生在backward之后，这是正确的
            
            # 但是，如果我们在adaptive_density_control中改变了参数，
            # 那么在下一次forward之前，我们需要重新初始化优化器
            
            # 这里的潜在问题是：Adam优化器内部维护了动量等状态，这些状态的形状必须与参数匹配
            # 当我们改变参数形状时，必须重新初始化优化器
            
            train_config = self.config['training']
            
            # 获取当前学习率（可能已经衰减）
            current_lr_dict = {}
            for param_group in self.optimizer.param_groups:
                # 假设参数组顺序固定
                pass
            
            # 简单起见，使用初始学习率或当前衰减后的学习率
            # 这里我们重新创建一个新的优化器
            
            params = self.gaussian_model.get_optimizable_params(
                lr_position=train_config.get('gaussian', {}).get('position_lr', 0.001),
                lr_density=train_config.get('gaussian', {}).get('density_lr', 0.01),
                lr_scale=train_config.get('gaussian', {}).get('scale_lr', 0.005),
                lr_rotation=train_config.get('gaussian', {}).get('rotation_lr', 0.001)
            )
            self.optimizer = optim.Adam(params)
            
            # 还需要更新scheduler
            self._setup_optimizer() # 重新设置optimizer和scheduler
            
            # 打印信息
            print(f"  Re-initialized optimizer with {self.gaussian_model.num_points} points")
        
        return stats
    
    def train_step(self) -> Dict[str, float]:
        """
        单步训练
        
        Returns:
            包含loss值的字典
        """
        self.gaussian_model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        volume, kspace_pred = self.forward()
        
        # 计算loss
        loss_dict = self.criterion(
            kspace_pred=kspace_pred,
            kspace_target=self.kspace_undersampled,  # 使用欠采样k-space作为目标
            mask=self.mask,
            image_pred=volume,
            image_target=self.target_image
        )
        
        # 反向传播
        loss_dict['total_loss'].backward()
        
        # 梯度裁剪
        max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.gaussian_model.parameters(),
                max_grad_norm
            )
        
        # 更新参数
        self.optimizer.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估当前重建质量
        
        Returns:
            评估指标字典
        """
        self.gaussian_model.eval()
        
        with torch.no_grad():
            volume, kspace_pred = self.forward()
            
            # 计算评估指标
            metrics = evaluate_reconstruction(
                pred=volume,
                target=self.target_image,
                compute_3d_ssim=True
            )
        
        return metrics
    
    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'gaussian_state': self.gaussian_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        
        # 保存最新checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
        
        # 定期保存
        save_interval = self.config['output'].get('save_interval', 500)
        if iteration % save_interval == 0:
            iter_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_{iteration:06d}.pth'
            )
            torch.save(checkpoint, iter_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.current_iteration = checkpoint['iteration']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        self.best_ssim = checkpoint.get('best_ssim', 0.0)
        
        print(f"Loaded checkpoint from iteration {self.current_iteration}")
    
    def save_reconstruction(self, iteration: int):
        """保存重建结果"""
        self.gaussian_model.eval()
        
        with torch.no_grad():
            volume, kspace_pred = self.forward()
            
            # 转换为numpy
            volume_np = volume.detach().cpu().numpy()
            
            # 保存为.npy
            result_path = os.path.join(
                self.result_dir,
                f'reconstruction_{iteration:06d}.npy'
            )
            np.save(result_path, volume_np)
            
            # 保存最终结果
            final_path = os.path.join(self.result_dir, 'reconstruction_final.npy')
            np.save(final_path, volume_np)
    
    def train(self, resume_from: Optional[str] = None):
        """
        完整训练流程
        
        Args:
            resume_from: 恢复训练的checkpoint路径
        """
        train_config = self.config['training']
        max_iterations = train_config['max_iterations']
        eval_interval = train_config.get('eval_interval', 100)
        log_interval = train_config.get('log_interval', 10)
        
        # 恢复训练
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        start_iter = self.current_iteration
        
        print(f"\nStarting training from iteration {start_iter}")
        print(f"Total iterations: {max_iterations}")
        print("-" * 50)
        
        # 训练循环
        pbar = tqdm(
            range(start_iter, max_iterations),
            desc="Training",
            dynamic_ncols=True
        )
        
        for iteration in pbar:
            self.current_iteration = iteration
            start_time = time.time()
            
            # 训练步骤
            # 注意：如果上一步进行了densification，optimizer已经被重新初始化
            # 这里的train_step会使用新的optimizer
            loss_dict = self.train_step()
            
            # 自适应密度控制
            # 这会改变模型参数形状，并重新初始化optimizer
            # 必须在train_step之后调用，这样下一次迭代才会使用新的optimizer
            adaptive_stats = self.adaptive_density_control(iteration)
            
            # 学习率调度
            self.scheduler.step()
            
            # 日志
            if iteration % log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'kspace': f"{loss_dict['kspace_loss']:.4f}",
                    'n_pts': self.gaussian_model.num_points
                })
            
            # 评估
            if iteration % eval_interval == 0 or iteration == max_iterations - 1:
                metrics = self.evaluate()
                
                is_best = metrics['psnr'] > self.best_psnr
                if is_best:
                    self.best_psnr = metrics['psnr']
                    self.best_ssim = metrics['ssim']
                
                print(f"\n[Iter {iteration}] "
                      f"PSNR: {metrics['psnr']:.2f} dB, "
                      f"SSIM: {metrics['ssim']:.4f}, "
                      f"NMSE: {metrics['nmse']:.6f}")
                
                if adaptive_stats['split'] > 0 or adaptive_stats['clone'] > 0:
                    print(f"  Density control: split={adaptive_stats['split']}, "
                          f"clone={adaptive_stats['clone']}, "
                          f"prune={adaptive_stats['prune']}")
                
                print(f"  Num Gaussians: {self.gaussian_model.num_points}")
                
                # 保存checkpoint
                self.save_checkpoint(iteration, is_best)
        
        # 训练结束
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        
        # 保存最终结果
        self.save_reconstruction(max_iterations)
        
        return self.best_psnr, self.best_ssim
