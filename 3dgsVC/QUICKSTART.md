# 快速启动指南

## 环境配置

```bash
# 创建虚拟环境
conda create -n 3dgsmr python=3.10 -y
conda activate 3dgsmr

# 安装依赖
cd /data/data54/wanghaobo/3DGS/3dgsVC
pip install -r requirements.txt
```

## 训练

### 方式1: 使用shell脚本 (推荐)

```bash
# 默认配置训练 (4x加速, 500初始高斯点, 2000次迭代)
cd /data/data54/wanghaobo/3DGS/3dgsVC
bash scripts/train.sh

# 指定参数
bash scripts/train.sh --acceleration 6 --gpu 0 --max_iterations 3000 --initial_points 800

# 恢复训练
bash scripts/train.sh --resume outputs/acc4_pts500_seed42/checkpoints/latest.pth
```

### 方式2: 直接使用Python

```bash
python train.py --config configs/default.yaml \
    --data_path /data/data54/wanghaobo/data/ksp_full.h5 \
    --output_dir ./outputs \
    --gpu 0 \
    --acceleration 4 \
    --max_iterations 2000 \
    --initial_points 500
```

## 测试/推理

### 方式1: 使用shell脚本

```bash
# 测试训练好的模型
bash scripts/test.sh --checkpoint outputs/acc4_pts500_seed42/checkpoints/best.pth

# 保存切片图像
bash scripts/test.sh --checkpoint outputs/acc4_pts500_seed42/checkpoints/best.pth --save_slices

# 自定义输出目录
bash scripts/test.sh --checkpoint outputs/acc4_pts500_seed42/checkpoints/best.pth \
    --output_dir my_results/
```

### 方式2: 直接使用Python

```bash
python test.py --checkpoint outputs/acc4_pts500_seed42/checkpoints/best.pth \
    --data_path /data/data54/wanghaobo/data/ksp_full.h5 \
    --output_dir test_results/ \
    --save_volume \
    --save_slices
```

## 配置文件修改

编辑 `configs/default.yaml`:

```yaml
# 关键参数
gaussian:
  initial_num_points: 500      # 初始高斯点数
  init_scale: 2.0               # 初始尺度
  
training:
  max_iterations: 2000          # 最大迭代数
  lr_position: 0.0001           # 位置学习率
  lr_density: 0.001             # 密度学习率
  
adaptive_control:
  long_axis_splitting: true     # 使用长轴分裂(对高加速因子有效)
  enable_cloning: true          # 使用克隆
  densify_start: 100            # 开始密度控制
  densify_end: 1500             # 结束密度控制
```

## 输出文件说明

训练输出:
```
outputs/
├── config.yaml                 # 保存的配置
├── checkpoints/
│   ├── latest.pth             # 最新checkpoint
│   ├── best.pth               # 最佳checkpoint
│   └── checkpoint_*.pth       # 定期保存
└── results/
    └── reconstruction_final.npy  # 最终重建
```

测试输出:
```
test_results/
├── metrics.yaml                # PSNR, SSIM, NMSE
├── reconstruction.npy          # 重建体积(complex)
├── reconstruction_magnitude.npy # 幅度图
├── target.npy                  # 目标图像
├── zero_filled.npy            # 零填充baseline
└── slices/                     # 切片图像 (可选)
```

## 常见问题

### Q: 内存不足?
A: 减少 `initial_num_points` 或使用批处理。修改config:
```yaml
gaussian:
  initial_num_points: 300
```

### Q: 训练很慢?
A: 这是正常的。3D Gaussian表示计算复杂。可以:
- 减少 `initial_num_points`
- 减少 `max_iterations`
- 使用更强的GPU

### Q: 重建质量不好?
A: 需要训练更多迭代:
- 增加 `max_iterations` (通常2000-5000)
- 增加 `initial_num_points` (通常500-2000)
- 调整学习率

### Q: 如何使用不同的加速因子?
A: 使用命令行参数:
```bash
bash scripts/train.sh --acceleration 8
```

高加速因子(6x, 8x)需要:
- 更多迭代
- 启用长轴分裂 (已默认启用)
- 可能需要更多初始高斯点

## 论文信息

Title: Three-Dimensional MRI Reconstruction with Gaussian Representations: Tackling the Undersampling Problem

Key innovation: 使用各向异性3D高斯表示k-space欠采样MRI重建,通过自适应密度控制(长轴分裂、克隆、剪枝)处理不同加速因子。
