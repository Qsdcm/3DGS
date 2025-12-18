# 3DGSMR: 3D Gaussian Representation for MRI Reconstruction

基于论文 "Three-Dimensional MRI Reconstruction with Gaussian Representations: Tackling the Undersampling Problem" 的实现。

## 项目结构

```
3dgsVC/
├── configs/
│   └── default.yaml          # 默认配置文件
├── data/
│   ├── __init__.py
│   ├── dataset.py            # 数据加载器
│   └── transforms.py         # FFT变换
├── gaussian/
│   ├── __init__.py
│   ├── gaussian_model.py     # 3D Gaussian模型
│   └── voxelizer.py          # Gaussian→体素渲染
├── losses/
│   ├── __init__.py
│   └── losses.py             # 损失函数
├── metrics/
│   ├── __init__.py
│   └── metrics.py            # 评估指标 (PSNR, SSIM, NMSE)
├── trainers/
│   ├── __init__.py
│   └── trainer.py            # 训练器
├── scripts/
│   ├── train.sh              # 训练脚本
│   └── test.sh               # 测试脚本
├── train.py                  # 训练入口
├── test.py                   # 测试入口
└── README.md
```

## 算法原理

### 核心思想
使用3D各向异性高斯表示MRI体积信号，通过优化高斯参数重建欠采样的MRI数据。

### 关键公式

**体素渲染 (论文公式3)**:
$$x_j = \sum_i G_i^3(j|\rho_i, p_i, \Sigma_i)$$

其中:
- $\rho_i$: 复数密度 (complex density)
- $p_i$: 位置 (position)
- $\Sigma_i = R S S^T R^T$: 协方差矩阵 (通过旋转和缩放构建)

**前向模型**:
$$\text{k-space} = \text{FFT}(\text{Gaussians} \rightarrow \text{Volume})$$

**损失函数 (论文公式2)**:
$$X^* = \arg\min_X \|A(X) - b\|^2 + \lambda R(X)$$

其中 $A(X)$ 是采样后的k-space，$b$ 是观测值。

### 自适应密度控制

论文Section IV提出的关键策略:
1. **长轴分裂 (Long-axis Splitting)**: 沿高斯最长轴分裂，对高加速因子(6x, 8x)特别有效
2. **克隆 (Clone)**: 复制小尺度高梯度高斯
3. **剪枝 (Prune)**: 移除低密度或过大高斯

## 环境配置

```bash
# 创建虚拟环境
conda create -n 3dgsmr python=3.10 -y
conda activate 3dgsmr

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy h5py pyyaml tqdm scikit-image matplotlib
```

## 数据格式

支持 HDF5 格式的 k-space 数据:
- 文件结构: `{'kspace': (num_coils, D, H, W)}` 或 `{'kspace': (D, H, W)}`
- 数据类型: complex64 或 complex128

## 使用方法

### 训练

```bash
# 使用默认配置
bash scripts/train.sh

# 指定参数
bash scripts/train.sh --acceleration 4 --gpu 0 --max_iterations 2000

# 使用不同加速因子
bash scripts/train.sh --acceleration 6

# 从checkpoint恢复
bash scripts/train.sh --resume outputs/checkpoints/latest.pth
```

命令行参数:
- `--config`: 配置文件路径
- `--acceleration`: 加速因子 (2, 4, 6, 8)
- `--gpu`: GPU设备ID
- `--max_iterations`: 最大迭代次数
- `--initial_points`: 初始高斯点数
- `--seed`: 随机种子
- `--resume`: 恢复训练的checkpoint

### 测试

```bash
# 基本测试
bash scripts/test.sh --checkpoint outputs/checkpoints/best.pth

# 保存切片图像
bash scripts/test.sh --checkpoint outputs/checkpoints/best.pth --save_slices

# 指定输出目录
bash scripts/test.sh --checkpoint outputs/checkpoints/best.pth --output_dir my_results/
```

### Python API

```python
from trainers import GaussianTrainer
import yaml

# 加载配置
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# 创建训练器
trainer = GaussianTrainer(config=config)

# 训练
trainer.train()

# 评估
metrics = trainer.evaluate()
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
```

## 配置说明

`configs/default.yaml` 主要配置项:

```yaml
data:
  data_path: /path/to/ksp_full.h5    # 数据路径
  acceleration_factor: 4              # 加速因子
  mask_type: gaussian                 # 采样mask类型
  center_fraction: 0.08               # 中心保留比例

gaussian:
  initial_num_points: 500             # 初始高斯点数
  init_scale: 0.5                     # 初始缩放
  max_num_points: 50000               # 最大高斯点数

adaptive_control:
  long_axis_splitting: true           # 使用长轴分裂
  enable_cloning: true                # 使用克隆
  densify_start: 100                  # 开始密度控制
  densify_end: 1500                   # 结束密度控制
  densify_interval: 100               # 控制间隔

training:
  max_iterations: 2000                # 最大迭代次数
  lr_position: 0.0001                 # 位置学习率
  lr_density: 0.001                   # 密度学习率
```

## 输出文件

训练输出:
```
outputs/
├── config.yaml               # 保存的配置
├── checkpoints/
│   ├── latest.pth           # 最新checkpoint
│   ├── best.pth             # 最佳checkpoint
│   └── checkpoint_*.pth     # 定期保存
└── results/
    └── reconstruction_final.npy  # 最终重建结果
```

测试输出:
```
test_results/
├── metrics.yaml              # 评估指标
├── reconstruction.npy        # 重建体积 (complex)
├── reconstruction_magnitude.npy  # 幅度图
├── target.npy                # 目标
├── zero_filled.npy           # 零填充baseline
└── slices/                   # 切片图像 (可选)
```

## 论文引用

```bibtex
@article{3dgsmr2025,
  title={Three-Dimensional MRI Reconstruction with Gaussian Representations: Tackling the Undersampling Problem},
  author={...},
  journal={arXiv preprint arXiv:2502.06510},
  year={2025}
}
```

## 注意事项

1. **内存**: 高斯点数较多时GPU内存消耗较大，可减少`initial_num_points`或`max_num_points`
2. **收敛**: 通常1000-2000次迭代可收敛，可通过观察PSNR判断
3. **加速因子**: 高加速因子(6x, 8x)需要更多迭代和更精细的密度控制
4. **初始化**: 从零填充重建初始化通常优于随机初始化
