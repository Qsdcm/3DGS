# 3D Gaussian Splatting for MRI Reconstruction (3DGS-MRI)

基于 3D Gaussian Splatting 的 MRI 重建方法。从欠采样 k-space 数据重建 3D 复数体数据。

## 概述

本项目实现了一种新颖的 MRI 重建方法，使用 3D Gaussian Splatting (3DGS) 作为隐式表示来重建欠采样的 MRI 数据。

### 核心特点

1. **3DGS 表示**：使用 M 个 3D Gaussian 点来表示 3D 复数体数据
   - 每个 Gaussian 包含：center (位置)、scale (尺度)、rotation (旋转，baseline 固定为 identity)、rho (复数密度)

2. **可微分体素化**：将 Gaussians 体素化为 3D complex volume，支持反向传播

3. **MRI Forward Operator**：A(X) = mask ⊙ FFT3(X)，使用与数据读取一致的 FFT 约定

4. **自适应控制**：支持 densification (分裂) 和 pruning (剪枝) 以优化 Gaussian 分布

5. **分布式支持**：支持单机多卡并行重建不同样本

## 数据格式

### H5 文件要求

- **文件格式**：HDF5 (.h5 或 .hdf5)
- **必需 dataset**：`kspace`
- **kspace shape**：`(nc, nz, nx, ny)`，其中 nc 为线圈数
- **数据类型**：complex64 或 complex128

示例数据结构：
```
file.h5
└── kspace: (8, 64, 128, 128) complex64
    # 8 线圈, 64 slices, 128x128 矩阵
```

### 数据处理流程

1. 加载全采样 k-space：`(nc, nz, nx, ny) complex`
2. 生成 3D 欠采样 mask：`(nz, nx, ny) float`
3. 应用欠采样：`kspace_under = kspace_full * mask`
4. **单通道近似**：`kspace_under_single = mean(kspace_under, dim=0)`
5. GT 图像：多线圈 iFFT 后 SoS 合并

> **注意**：当前实现使用单通道均值近似。严格的多线圈重建需要 coil sensitivity maps (CSM)。

## 安装

```bash
# 克隆项目
git clone <repo_url>
cd 3dgsmri

# 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install torch numpy h5py scikit-image
```

## 快速开始

### 单 GPU 训练

```bash
# 设置变量
GPU=0
DATA=/path/to/your/data.h5  # 或包含 H5 文件的目录
OUT=./outputs
ITERS=2000

# 运行训练
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --data_root $DATA \
    --out_root $OUT \
    --max_iters $ITERS \
    --acceleration 4.0 \
    --mask_type gaussian
```

或使用脚本：

```bash
GPU=0 DATA=/path/to/data OUT=./outputs ITERS=2000 bash train.sh
```

### 多 GPU 训练

```bash
# 使用 4 张 GPU
GPU=0,1,2,3
CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=4 train.py \
    --data_root $DATA \
    --out_root $OUT \
    --max_iters 2000 \
    --distributed 1
```

或使用脚本：

```bash
GPU=0,1,2,3 NGPU=4 MULTI_GPU=1 DATA=/path/to/data OUT=./outputs bash train.sh
```

### 评估

```bash
# 基本评估（使用保存的 metrics）
python test.py --out_root ./outputs

# 重新计算指标（需要原始数据）
python test.py --data_root /path/to/data --out_root ./outputs
```

## 输出结构

训练完成后，每个样本的结果保存在：

```
outputs/
├── sample_name/
│   ├── recon_complex.npy    # 重建的复数图像 (nz, nx, ny) complex64
│   ├── recon_mag.npy        # 重建的幅值图像 (nz, nx, ny) float32
│   ├── gaussians.pt         # 最终 Gaussian 参数
│   └── metrics.json         # 评估指标和训练信息
├── summary.json             # 汇总统计
└── results.json             # 评估结果（test.py 生成）
```

## 评估指标

- **PSNR** (dB)：峰值信噪比，越高越好
- **SSIM**：结构相似性，范围 [0, 1]，越高越好
- **NMSE**：归一化均方误差，越低越好

> **计算方式**：在 magnitude 空间对比 `abs(recon_complex)` 与 `image_gt`。SSIM 按 slice 计算后取平均。

## 参数说明

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_gaussians` | 1000 | 初始 Gaussian 数量 |
| `--max_iters` | 2000 | 每个样本的优化迭代次数 |
| `--lr_centers` | 0.001 | 中心点学习率 |
| `--lr_scales` | 0.005 | 尺度学习率 |
| `--lr_rho` | 0.01 | 密度学习率 |
| `--acceleration` | 4.0 | 欠采样加速倍数 |
| `--mask_type` | gaussian | mask 类型 (gaussian/uniform) |

### 自适应控制

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--densify_every` | 100 | densification 间隔 |
| `--densify_start` | 100 | 开始 densify 的迭代 |
| `--densify_end` | 1500 | 停止 densify 的迭代 |
| `--grad_threshold` | 0.01 | 分裂梯度阈值 |
| `--prune_rho_thresh` | 0.001 | 剪枝密度阈值 |
| `--max_gaussians` | 50000 | Gaussian 数量上限 |

### 正则化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_tv` | 0.0 | TV 正则化权重（0 禁用）|

## 技术细节

### Gaussian 参数化

每个 Gaussian i 的参数：
- `center p_i ∈ R³`：中心位置，范围 [-1, 1]
- `log_scale s_i ∈ R³`：对数尺度（实际 scale = exp(log_scale)）
- `rho_i ∈ C`：复数密度，用两个 float (real, imag) 表示
- `rotation q_i ∈ R⁴`：四元数（baseline 固定为 identity）

### 体素化公式

```
X_hat(v) = Σ_i ρ_i * exp(-0.5 * (x - p_i)ᵀ Σ_i⁻¹ (x - p_i))
```

其中 `Σ_i = diag(scale_i²)` (axis-aligned, 无旋转)。

### 损失函数

```
L = L_dc + λ_tv * L_tv

L_dc = mean(|mask ⊙ (FFT3(X_hat) - b)|²)
L_tv = TV(|X_hat|)
```

### 初始化策略

1. 从零填充图像 `image_init_complex` 提取背景过滤后的高能体素
2. 随机采样 M 个点作为 Gaussian 中心
3. `rho` 初始化为采样点的复数值
4. `scale` 初始化为到 k 个最近邻的平均距离

### Long-axis Splitting

当 Gaussian 的中心梯度超过阈值时：
1. 找到 scale 最大的轴 k
2. 沿该轴方向分裂为两个新 Gaussian：`p± = p ± δ * e_k`
3. scale 缩小，rho 平分

## 性能说明

### Baseline Voxelizer

当前使用纯 PyTorch 实现的 baseline voxelizer：
- 对每个 Gaussian 计算其 3-sigma bbox 内的贡献
- 避免创建 (M, nz, nx, ny) 级别的大张量
- 适合中等规模问题（~50k Gaussians, ~128³ volume）

**性能提示**：
- 对于大规模问题，考虑实现 CUDA 加速的 voxelizer
- 增加 `sigma_cutoff` 可减少计算量但可能影响精度
- 适当控制 `max_gaussians` 避免显存溢出

### 分布式策略

- 并行重建不同样本（数据并行）
- 每个 rank 处理 data_root 下不同 H5 文件子集
- 不需要跨 rank 同步梯度（每个 volume 独立优化）

## 扩展接口

### 自定义 Voxelizer

```python
class CustomVoxelizer(nn.Module):
    def forward(self, centers, scales, rho_real, rho_imag, rotations=None):
        # 实现自定义体素化逻辑
        # 返回 (nz, nx, ny) complex volume
        pass
```

### 启用旋转

设置 `use_rotation=True` 并实现四元数到旋转矩阵的转换：

```python
model = GaussianMRIModel(
    volume_shape=shape,
    use_rotation=True,  # 启用旋转
    ...
)
```

## 常见问题

**Q: 显存不足？**
- 减少 `n_gaussians` 和 `max_gaussians`
- 减小 batch size（voxelizer 中的 `batch_size`）
- 使用更小的 volume

**Q: 重建质量差？**
- 增加 `max_iters`
- 调整学习率
- 增加 `n_gaussians`
- 检查数据归一化

**Q: 训练不稳定？**
- 降低学习率
- 调整 `densify_every` 和相关阈值
- 禁用 densification 先验证基本流程

## 许可证

[Your License Here]

## 引用

如果您使用本项目，请引用：

```bibtex
@misc{3dgsmri,
  title={3D Gaussian Splatting for MRI Reconstruction},
  year={2024}
}
```
