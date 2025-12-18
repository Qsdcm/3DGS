"""
3DGSMR Training Entry Point

用法:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/latest.pth

论文: Three-Dimensional MRI Reconstruction with Gaussian Representations:
      Tackling the Undersampling Problem
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random

from trainers import GaussianTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train 3DGSMR for MRI Reconstruction'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Override data path in config'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--acceleration',
        type=int,
        default=None,
        help='Override acceleration factor'
    )
    
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=None,
        help='Override max iterations'
    )
    
    parser.add_argument(
        '--initial_points',
        type=int,
        default=None,
        help='Override initial number of Gaussian points'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args) -> dict:
    """使用命令行参数覆盖配置"""
    if args.data_path is not None:
        config['data']['data_path'] = args.data_path
    
    if args.output_dir is not None:
        config['output']['output_dir'] = args.output_dir
    
    if args.acceleration is not None:
        config['data']['acceleration_factor'] = args.acceleration
    
    if args.max_iterations is not None:
        config['training']['max_iterations'] = args.max_iterations
    
    if args.initial_points is not None:
        config['gaussian']['initial_num_points'] = args.initial_points
    
    return config


def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # 加载配置
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    config = override_config(config, args)
    
    # 打印关键配置
    print("\n" + "=" * 50)
    print("3DGSMR Training Configuration")
    print("=" * 50)
    print(f"Data path: {config['data']['data_path']}")
    print(f"Acceleration factor: {config['data']['acceleration_factor']}")
    print(f"Initial Gaussians: {config['gaussian']['initial_num_points']}")
    print(f"Max iterations: {config['training']['max_iterations']}")
    print(f"Output directory: {config['output']['output_dir']}")
    print("=" * 50 + "\n")
    
    # 创建Trainer
    trainer = GaussianTrainer(config=config, device=device)
    
    # 开始训练
    best_psnr, best_ssim = trainer.train(resume_from=args.resume)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Best SSIM: {best_ssim:.4f}")
    print(f"Results saved to: {config['output']['output_dir']}")
    print("=" * 50)


if __name__ == '__main__':
    main()
