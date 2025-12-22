"""
3DGSMR Testing/Inference Entry Point (Enhanced)

功能:
1. 加载训练好的模型 (支持指定权重)
2. 执行重建 (支持指定加速倍数)
3. 自动保存三组数据: 原图(GT), 欠采样(ZF), 重建(Recon)
4. 生成对比切片图像
5. 自动根据加速倍数生成输出文件夹
"""

import os
import random
import argparse
import yaml
import torch
import numpy as np
import time
from typing import Dict, Any

# 尝试导入 nibabel 用于保存 .nii 文件
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from data import MRIDataset
from gaussian import GaussianModel3D, Voxelizer
from metrics import evaluate_reconstruction, print_metrics

def set_seed(seed: int):
    """设置随机种子 (复制自 train.py)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Test 3DGSMR for MRI Reconstruction')
    
    # --- 修改部分：根据需求调整了参数 ---
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset (e.g., /data/datasets/brain)')
    parser.add_argument('--weights', type=str, required=True, help='Path to checkpoint/weights file (.pt/.pth)')
    parser.add_argument('--acceleration', type=float, default=1.0, help='Acceleration factor (e.g., 4, 8)')
    
    # 其他可选参数
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save_volume', action='store_true', default=True, help='Save volumes as .npy/.nii')
    parser.add_argument('--save_slices', action='store_true', default=True, help='Save slice comparison images')
    
    return parser.parse_args()

class GaussianTester:
    def __init__(self, checkpoint_path: str, config: Dict[str, Any] = None, device: torch.device = None, 
                 data_path: str = None, acceleration_override: float = None):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.acceleration_override = acceleration_override
        
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理 Config
        if config is not None:
            self.config = config
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint and not provided as argument.")
        
        # 如果命令行指定了 dataset 路径，覆盖 config
        if data_path is not None:
            self.config['data']['data_path'] = data_path
        
        self._setup_data()
        self._setup_model(checkpoint)
        
        print(f"Loaded model with {self.gaussian_model.num_points} Gaussians")
        
    def _setup_data(self):
        data_config = self.config['data']
        
        # --- 修改部分：优先使用命令行传入的加速倍数 ---
        acc_factor = self.acceleration_override if self.acceleration_override is not None else data_config['acceleration_factor']
        
        print(f"Loading data from: {data_config['data_path']}")
        print(f"Using Acceleration Factor: {acc_factor}x")
        
        self.dataset = MRIDataset(
            data_path=data_config['data_path'],
            acceleration_factor=acc_factor,  # 使用覆盖后的倍数
            mask_type=data_config.get('mask_type', 'gaussian'),
            use_acs=data_config.get('use_acs', True),
            acs_lines=int(data_config.get('center_fraction', 0.08) * 256)
        )
        
        data = self.dataset.get_data()
        self.volume_shape = data['volume_shape']
        self.target_image = data['ground_truth'].to(self.device) # 原图
        self.zero_filled = data['zero_filled'].to(self.device)   # 欠采样(零填充)
        
    def _setup_model(self, checkpoint: Dict):
        gaussian_state = checkpoint['gaussian_state']
        num_points = gaussian_state['positions'].shape[0]
        
        self.gaussian_model = GaussianModel3D(
            num_points=num_points,
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        self.gaussian_model.load_state_dict(checkpoint['gaussian_state'])
        
        # 使用最新的 Voxelizer
        self.voxelizer = Voxelizer(
            volume_shape=tuple(self.volume_shape),
            device=str(self.device)
        )
        
    def reconstruct(self) -> torch.Tensor:
        self.gaussian_model.eval()
        with torch.no_grad():
            t0 = time.time()
            volume = self.voxelizer(
                positions=self.gaussian_model.positions,
                scales=self.gaussian_model.get_scales(),
                rotations=self.gaussian_model.rotations,
                density=self.gaussian_model.get_densities()
            )
            print(f"Reconstruction took {time.time()-t0:.2f}s")
        return volume
    
    def save_results(self, output_dir: str, save_volume: bool = True, save_slices: bool = True):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running reconstruction...")
        recon_volume = self.reconstruct()
        
        # 计算指标
        metrics = evaluate_reconstruction(recon_volume, self.target_image, compute_3d_ssim=True)
        zf_metrics = evaluate_reconstruction(self.zero_filled, self.target_image, compute_3d_ssim=True)
        
        print("\n" + "="*50)
        print("  Final Results")
        print("="*50)
        print(f"Original (GT) vs Zero-Filled (Input) vs 3DGSMR (Output)")
        print(f"PSNR: {zf_metrics['psnr']:.2f} -> {metrics['psnr']:.2f} dB")
        print(f"SSIM: {zf_metrics['ssim']:.4f} -> {metrics['ssim']:.4f}")
        print("="*50)
        
        # 保存指标
        with open(os.path.join(output_dir, 'metrics.yaml'), 'w') as f:
            yaml.dump({'3dgsmr': metrics, 'zero_filled': zf_metrics}, f)

        # 准备保存的数据 (转换为 numpy)
        recon_np = recon_volume.detach().cpu().numpy()
        target_np = self.target_image.detach().cpu().numpy()
        zf_np = self.zero_filled.detach().cpu().numpy()
        
        if save_volume:
            print(f"\nSaving volumes to {output_dir} ...")
            # 1. 保存为 .npy (复数数据)
            np.save(os.path.join(output_dir, 'reconstruction.npy'), recon_np)
            np.save(os.path.join(output_dir, 'target.npy'), target_np)
            np.save(os.path.join(output_dir, 'zero_filled.npy'), zf_np)
            
            # 2. 保存为 .nii.gz (幅度图，医学通用格式)
            if HAS_NIBABEL:
                self._save_nifti(np.abs(recon_np), os.path.join(output_dir, 'reconstruction.nii.gz'))
                self._save_nifti(np.abs(target_np), os.path.join(output_dir, 'target.nii.gz'))
                self._save_nifti(np.abs(zf_np), os.path.join(output_dir, 'zero_filled.nii.gz'))
                print("Saved .nii.gz files for visualization.")
        
        if save_slices:
            print("Generating comparison slices...")
            self._save_comparison_slices(target_np, zf_np, recon_np, output_dir)
            
    def _save_nifti(self, volume_abs, path):
        img = nib.Nifti1Image(volume_abs, np.eye(4))
        nib.save(img, path)

    def _save_comparison_slices(self, target, zf, recon, output_dir):
        """生成三图对比切片: GT | Zero-Filled | Recon"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        slice_dir = os.path.join(output_dir, 'slices')
        os.makedirs(slice_dir, exist_ok=True)
        
        t_mag = np.abs(target)
        z_mag = np.abs(zf)
        r_mag = np.abs(recon)
        
        vmax = np.percentile(t_mag, 99.9)
        D, H, W = t_mag.shape
        
        slices_map = {
            'Axial': (0, [D//3, D//2, 2*D//3]),
            'Coronal': (1, [H//3, H//2, 2*H//3]),
            'Sagittal': (2, [W//3, W//2, 2*W//3])
        }
        
        for view_name, (dim, idxs) in slices_map.items():
            for idx in idxs:
                if dim == 0:
                    imgs = [t_mag[idx,:,:], z_mag[idx,:,:], r_mag[idx,:,:]]
                elif dim == 1:
                    imgs = [t_mag[:,idx,:], z_mag[:,idx,:], r_mag[:,idx,:]]
                else:
                    imgs = [t_mag[:,:,idx], z_mag[:,:,idx], r_mag[:,:,idx]]
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                titles = ['Original (GT)', 'Undersampled (ZF)', 'Reconstruction']
                
                for ax, img, title in zip(axes, imgs, titles):
                    ax.imshow(img, cmap='gray', vmin=0, vmax=vmax)
                    ax.set_title(title)
                    ax.axis('off')
                
                plt.suptitle(f"{view_name} Slice {idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(slice_dir, f'{view_name}_slice_{idx:03d}.png'), dpi=150)
                plt.close()
        print(f"Saved slice comparisons to {slice_dir}")

def main():
    set_seed(42)  
    print(f"Random Seed set to: 42 (Synced with Training)")
    args = parse_args()
    
    # --- 修改部分：路径构建逻辑 ---
    # 1. 路径
    base_project_path = "/data/data54/wanghaobo/3DGS/3dgsVC"
    
    # 2. 处理加速倍数显示 (2.0 -> 2, 2.5 -> 2.5)
    acc_tag = int(args.acceleration) if args.acceleration.is_integer() else args.acceleration
    
    # 3. 自动生成输出文件夹名
    output_folder_name = f"test_results_{acc_tag}x"
    save_dir = os.path.join(base_project_path, output_folder_name)
    
    # 打印配置确认
    print("-" * 40)
    print(f"Start Testing...")
    print(f"Dataset      : {args.dataset}")
    print(f"Weights      : {args.weights}")
    print(f"Acceleration : {args.acceleration}x")
    print(f"Output Dir   : {save_dir}")
    print("-" * 40)

    # 设备设置
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
    # 初始化 Tester (传入新的 dataset 和 acceleration 参数)
    tester = GaussianTester(
        checkpoint_path=args.weights,  # 使用 weights 参数
        config=config, 
        device=device, 
        data_path=args.dataset,        # 使用 dataset 参数
        acceleration_override=args.acceleration # 传入加速倍数
    )
    
    # 执行保存
    tester.save_results(save_dir, args.save_volume, args.save_slices)

if __name__ == '__main__':
    main()