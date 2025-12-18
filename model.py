"""model.py

3D Gaussian Splatting for MRI Reconstruction 模型模块。

核心组件：
- GaussianMRIModel: 封装 Gaussian 参数、体素化、densify/prune、forward operator
- Voxelizer: 将 Gaussians 体素化为 3D complex volume（baseline 纯 PyTorch 实现）
- MRI Forward Operator: A(X) = mask ⊙ FFT3(X)

每个 Gaussian i 参数：
- center p_i ∈ R^3（[-1, 1] 归一化坐标）
- log_scale s_i ∈ R^3（实际 scale = exp(log_scale)）
- rotation q_i ∈ R^4（quaternion，baseline 固定为 identity）
- rho_i ∈ C（complex density，用 rho_real + rho_imag 两个 float 参数化）
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np

from data_read import image_to_kspace


class Voxelizer(nn.Module):
    """将 Gaussians 体素化为 3D complex volume。

    Baseline 实现：纯 PyTorch，对每个 Gaussian 计算其 3-sigma bbox 内的贡献。

    体素化公式：
    X_hat(v) = sum_i rho_i * exp(-0.5 * (x - p_i)^T Σ_i^{-1} (x - p_i))

    当前实现：axis-aligned covariance（无旋转）
    Σ_i = diag(scale_i^2)
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        device: str = "cuda",
        sigma_cutoff: float = 3.0,
        use_rotation: bool = False,  # 预留接口，baseline 不启用
    ):
        """
        Args:
            volume_shape: (nz, nx, ny) 目标 volume 形状
            device: 目标设备
            sigma_cutoff: 截断范围（几倍 sigma）
            use_rotation: 是否启用旋转（当前 baseline 为 False）
        """
        super().__init__()
        self.nz, self.nx, self.ny = volume_shape
        self.device = device
        self.sigma_cutoff = sigma_cutoff
        self.use_rotation = use_rotation

        # 预计算归一化坐标网格
        # 坐标范围 [-1, 1]
        z = torch.linspace(-1, 1, self.nz, device=device)
        x = torch.linspace(-1, 1, self.nx, device=device)
        y = torch.linspace(-1, 1, self.ny, device=device)

        # 体素间距（在 [-1, 1] 空间中）
        self.voxel_size_z = 2.0 / (self.nz - 1) if self.nz > 1 else 2.0
        self.voxel_size_x = 2.0 / (self.nx - 1) if self.nx > 1 else 2.0
        self.voxel_size_y = 2.0 / (self.ny - 1) if self.ny > 1 else 2.0

        # 存储坐标轴（不需要完整网格，节省显存）
        self.register_buffer("coord_z", z)
        self.register_buffer("coord_x", x)
        self.register_buffer("coord_y", y)

    def forward(
        self,
        centers: torch.Tensor,  # (M, 3)
        scales: torch.Tensor,   # (M, 3) 实际 scale（已 exp）
        rho_real: torch.Tensor, # (M,)
        rho_imag: torch.Tensor, # (M,)
        rotations: Optional[torch.Tensor] = None,  # (M, 4) quaternion，当前不使用
    ) -> torch.Tensor:
        """体素化 Gaussians 为 3D complex volume。

        Args:
            centers: (M, 3) Gaussian 中心坐标（[-1, 1] 空间）
            scales: (M, 3) 各轴 scale（正数）
            rho_real: (M,) complex density 实部
            rho_imag: (M,) complex density 虚部
            rotations: (M, 4) quaternion（当前 baseline 忽略）

        Returns:
            volume: (nz, nx, ny) complex64 张量
        """
        M = centers.shape[0]

        # 初始化输出 volume
        volume_real = torch.zeros(self.nz, self.nx, self.ny, device=self.device)
        volume_imag = torch.zeros(self.nz, self.nx, self.ny, device=self.device)

        # 对每个 Gaussian，计算其 bbox 内的贡献
        # 为了避免 for 循环太慢，我们按 batch 处理
        # 但为了避免显存爆炸，每次处理一定数量的 Gaussians

        batch_size = min(M, 64)  # 每批处理的 Gaussian 数量

        for batch_start in range(0, M, batch_size):
            batch_end = min(batch_start + batch_size, M)
            batch_indices = torch.arange(batch_start, batch_end, device=self.device)

            self._process_gaussian_batch(
                centers[batch_indices],
                scales[batch_indices],
                rho_real[batch_indices],
                rho_imag[batch_indices],
                volume_real,
                volume_imag,
            )

        return torch.complex(volume_real, volume_imag)

    def _process_gaussian_batch(
        self,
        centers: torch.Tensor,  # (B, 3)
        scales: torch.Tensor,   # (B, 3)
        rho_real: torch.Tensor, # (B,)
        rho_imag: torch.Tensor, # (B,)
        volume_real: torch.Tensor,  # (nz, nx, ny) 累加目标
        volume_imag: torch.Tensor,  # (nz, nx, ny) 累加目标
    ):
        """处理一批 Gaussians，累加到 volume。"""
        B = centers.shape[0]

        for i in range(B):
            center = centers[i]  # (3,)
            scale = scales[i]    # (3,)
            rho_r = rho_real[i]
            rho_i = rho_imag[i]

            # 计算 3-sigma bbox（在 [-1, 1] 空间）
            sigma = scale  # axis-aligned: scale 就是 sigma
            max_sigma = sigma.max().item()
            radius = self.sigma_cutoff * max_sigma

            # 转换为体素索引范围
            cz, cx, cy = center[0].item(), center[1].item(), center[2].item()

            # 坐标到索引映射：idx = (coord + 1) / 2 * (N - 1)
            def coord_to_idx(coord, N):
                return (coord + 1) / 2 * (N - 1)

            def idx_to_coord(idx, N):
                return idx / (N - 1) * 2 - 1

            # 中心索引
            cz_idx = coord_to_idx(cz, self.nz)
            cx_idx = coord_to_idx(cx, self.nx)
            cy_idx = coord_to_idx(cy, self.ny)

            # bbox 半径（体素数）
            rz = int(np.ceil(radius / self.voxel_size_z)) + 1
            rx = int(np.ceil(radius / self.voxel_size_x)) + 1
            ry = int(np.ceil(radius / self.voxel_size_y)) + 1

            # 索引范围
            z_start = max(0, int(cz_idx) - rz)
            z_end = min(self.nz, int(cz_idx) + rz + 1)
            x_start = max(0, int(cx_idx) - rx)
            x_end = min(self.nx, int(cx_idx) + rx + 1)
            y_start = max(0, int(cy_idx) - ry)
            y_end = min(self.ny, int(cy_idx) + ry + 1)

            if z_start >= z_end or x_start >= x_end or y_start >= y_end:
                continue

            # 提取局部坐标
            local_z = self.coord_z[z_start:z_end]  # (Lz,)
            local_x = self.coord_x[x_start:x_end]  # (Lx,)
            local_y = self.coord_y[y_start:y_end]  # (Ly,)

            # 构建局部网格
            LZ, LX, LY = torch.meshgrid(local_z, local_x, local_y, indexing="ij")

            # 计算到中心的距离（axis-aligned）
            dz = (LZ - center[0]) / (scale[0] + 1e-8)
            dx = (LX - center[1]) / (scale[1] + 1e-8)
            dy = (LY - center[2]) / (scale[2] + 1e-8)

            # Gaussian 权重：exp(-0.5 * (dz^2 + dx^2 + dy^2))
            dist_sq = dz**2 + dx**2 + dy**2
            weight = torch.exp(-0.5 * dist_sq)

            # 截断（3-sigma 外设为 0）
            weight = weight * (dist_sq <= self.sigma_cutoff**2).float()

            # 累加到全局 volume
            volume_real[z_start:z_end, x_start:x_end, y_start:y_end] += rho_r * weight
            volume_imag[z_start:z_end, x_start:x_end, y_start:y_end] += rho_i * weight


class GaussianMRIModel(nn.Module):
    """3D Gaussian Splatting for MRI Reconstruction 主模型。

    包含：
    - Gaussian 参数（centers, log_scales, rho_real, rho_imag, rotations）
    - Voxelizer
    - MRI forward operator
    - Densification / Pruning
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        n_gaussians: int = 1000,
        device: str = "cuda",
        sigma_cutoff: float = 3.0,
        use_rotation: bool = False,
    ):
        """
        Args:
            volume_shape: (nz, nx, ny) 目标 volume 形状
            n_gaussians: 初始 Gaussian 数量
            device: 目标设备
            sigma_cutoff: Gaussian 截断范围
            use_rotation: 是否启用旋转（baseline 为 False）
        """
        super().__init__()
        self.volume_shape = volume_shape
        self.nz, self.nx, self.ny = volume_shape
        self.device = device
        self.use_rotation = use_rotation

        # Voxelizer
        self.voxelizer = Voxelizer(
            volume_shape=volume_shape,
            device=device,
            sigma_cutoff=sigma_cutoff,
            use_rotation=use_rotation,
        )

        # 初始化 Gaussian 参数（将在 initialize_from_image 中设置）
        self._init_gaussian_params(n_gaussians)

    def _init_gaussian_params(self, n_gaussians: int):
        """初始化 Gaussian 参数为随机值。"""
        # Centers: [-1, 1] 范围
        centers = torch.rand(n_gaussians, 3, device=self.device) * 2 - 1
        self.centers = nn.Parameter(centers)

        # Log scales: 初始化为较小的值
        log_scales = torch.full((n_gaussians, 3), -3.0, device=self.device)
        self.log_scales = nn.Parameter(log_scales)

        # Complex density (real + imag)
        rho_real = torch.randn(n_gaussians, device=self.device) * 0.01
        rho_imag = torch.randn(n_gaussians, device=self.device) * 0.01
        self.rho_real = nn.Parameter(rho_real)
        self.rho_imag = nn.Parameter(rho_imag)

        # Rotations (quaternion): 固定为 identity（预留接口）
        # [w, x, y, z] 格式，identity = [1, 0, 0, 0]
        rotations = torch.zeros(n_gaussians, 4, device=self.device)
        rotations[:, 0] = 1.0
        if self.use_rotation:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)

    def initialize_from_image(
        self,
        image_init_complex: torch.Tensor,  # (nz, nx, ny) complex
        n_gaussians: int,
        percentile_thresh: float = 10.0,
        k_init: float = 1.0,
        seed: int = 42,
    ):
        """从初始复数图像初始化 Gaussian 参数。

        步骤：
        1. 背景过滤：从 abs(x0) 里按阈值保留高能体素
        2. 从保留体素中随机采样 M 个点作为 centers
        3. 初始化 rho = k_init * x0_at_point
        4. 初始化 scale 为到最近邻的平均距离
        5. rotation 固定为 identity

        Args:
            image_init_complex: (nz, nx, ny) 初始复数图像
            n_gaussians: Gaussian 数量
            percentile_thresh: 背景过滤阈值（百分位）
            k_init: density 初始化系数
            seed: 随机种子
        """
        torch.manual_seed(seed)

        # 获取 magnitude
        mag = torch.abs(image_init_complex)  # (nz, nx, ny)

        # 背景过滤：保留高于阈值的体素
        thresh = torch.quantile(mag.flatten(), percentile_thresh / 100.0)
        valid_mask = mag > thresh  # (nz, nx, ny) bool

        # 获取有效体素的索引
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # (N_valid, 3)

        if valid_indices.shape[0] < n_gaussians:
            print(f"[Model] Warning: only {valid_indices.shape[0]} valid voxels, "
                  f"using all of them instead of {n_gaussians}")
            n_gaussians = valid_indices.shape[0]

        # 随机采样
        perm = torch.randperm(valid_indices.shape[0], device=self.device)[:n_gaussians]
        sampled_indices = valid_indices[perm]  # (M, 3)

        # 转换为归一化坐标 [-1, 1]
        centers = torch.zeros(n_gaussians, 3, device=self.device)
        centers[:, 0] = sampled_indices[:, 0].float() / (self.nz - 1) * 2 - 1  # z
        centers[:, 1] = sampled_indices[:, 1].float() / (self.nx - 1) * 2 - 1  # x
        centers[:, 2] = sampled_indices[:, 2].float() / (self.ny - 1) * 2 - 1  # y

        # 获取采样点的复数值
        sampled_values = image_init_complex[
            sampled_indices[:, 0],
            sampled_indices[:, 1],
            sampled_indices[:, 2],
        ]  # (M,) complex

        # 初始化 rho
        rho_real = k_init * sampled_values.real
        rho_imag = k_init * sampled_values.imag

        # 初始化 scale：计算到 k 个最近邻的平均距离
        log_scales = self._compute_initial_scales(centers, k_neighbors=3)

        # 设置参数
        self.centers = nn.Parameter(centers)
        self.log_scales = nn.Parameter(log_scales)
        self.rho_real = nn.Parameter(rho_real)
        self.rho_imag = nn.Parameter(rho_imag)

        # Rotations: identity
        rotations = torch.zeros(n_gaussians, 4, device=self.device)
        rotations[:, 0] = 1.0
        if self.use_rotation:
            self.rotations = nn.Parameter(rotations)
        else:
            self.register_buffer("rotations", rotations)

        print(f"[Model] Initialized {n_gaussians} Gaussians from image")

    def _compute_initial_scales(
        self,
        centers: torch.Tensor,  # (M, 3)
        k_neighbors: int = 3,
    ) -> torch.Tensor:
        """计算初始 log_scale：到 k 个最近邻的平均距离。"""
        M = centers.shape[0]

        if M <= k_neighbors:
            # 如果点太少，使用默认值
            voxel_size = 2.0 / max(self.nz, self.nx, self.ny)
            default_scale = voxel_size * 2
            return torch.full((M, 3), math.log(default_scale), device=self.device)

        # 计算两两距离
        # 为避免显存问题，使用分块计算或近似
        if M > 5000:
            # 对大量点使用随机采样近似
            sample_size = min(1000, M)
            sample_idx = torch.randperm(M, device=self.device)[:sample_size]
            sample_centers = centers[sample_idx]

            dists = torch.cdist(centers, sample_centers)  # (M, sample_size)
            # 排除自身（如果在采样中）
            dists = dists + torch.eye(M, sample_size, device=self.device)[:M, :sample_size] * 1e10
        else:
            dists = torch.cdist(centers, centers)  # (M, M)
            # 排除自身
            dists = dists + torch.eye(M, device=self.device) * 1e10

        # 取 k 个最近邻的平均距离
        k = min(k_neighbors, dists.shape[1] - 1)
        knn_dists, _ = torch.topk(dists, k, dim=1, largest=False)
        avg_dist = knn_dists.mean(dim=1)  # (M,)

        # 避免过小的 scale
        min_scale = 2.0 / max(self.nz, self.nx, self.ny) * 0.5
        avg_dist = torch.clamp(avg_dist, min=min_scale)

        # 各向同性初始化
        log_scales = torch.log(avg_dist).unsqueeze(1).expand(-1, 3)

        return log_scales.clone()

    @property
    def scales(self) -> torch.Tensor:
        """获取实际 scale（exp(log_scale)）。"""
        return torch.exp(self.log_scales)

    @property
    def n_gaussians(self) -> int:
        """当前 Gaussian 数量。"""
        return self.centers.shape[0]

    def voxelize(self) -> torch.Tensor:
        """将当前 Gaussians 体素化为 3D complex volume。

        Returns:
            volume: (nz, nx, ny) complex64
        """
        return self.voxelizer(
            self.centers,
            self.scales,
            self.rho_real,
            self.rho_imag,
            self.rotations if self.use_rotation else None,
        )

    def forward_mri(
        self,
        volume: torch.Tensor,  # (nz, nx, ny) complex
        mask: torch.Tensor,    # (nz, nx, ny) float
    ) -> torch.Tensor:
        """MRI forward operator: A(X) = mask ⊙ FFT3(X)

        使用 data_read.image_to_kspace() 确保一致性。

        Args:
            volume: (nz, nx, ny) 复数图像
            mask: (nz, nx, ny) 采样 mask

        Returns:
            kspace_masked: (nz, nx, ny) complex
        """
        kspace = image_to_kspace(volume)
        return mask * kspace

    def forward(
        self,
        mask: torch.Tensor,  # (nz, nx, ny) float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：体素化 + MRI forward。

        Returns:
            volume: (nz, nx, ny) complex - 重建的复数图像
            kspace_pred: (nz, nx, ny) complex - 预测的欠采样 k-space
        """
        volume = self.voxelize()
        kspace_pred = self.forward_mri(volume, mask)
        return volume, kspace_pred

    def compute_loss(
        self,
        kspace_pred: torch.Tensor,   # (nz, nx, ny) complex
        kspace_target: torch.Tensor, # (nz, nx, ny) complex
        mask: torch.Tensor,          # (nz, nx, ny) float
        volume: Optional[torch.Tensor] = None,  # (nz, nx, ny) complex
        lambda_tv: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数。

        L = L_dc + lambda_tv * L_tv

        L_dc = mean(|mask ⊙ (kspace_pred - kspace_target)|^2)
        L_tv = TV(|volume|)

        Args:
            kspace_pred: 预测 k-space
            kspace_target: 目标 k-space
            mask: 采样 mask
            volume: 重建图像（用于 TV，如果 None 则不计算 TV）
            lambda_tv: TV 正则系数

        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        # Data consistency loss
        diff = mask * (kspace_pred - kspace_target)
        loss_dc = torch.mean(torch.abs(diff) ** 2)

        loss_dict = {"loss_dc": loss_dc.item()}

        # TV regularization
        loss_tv = torch.tensor(0.0, device=self.device)
        if lambda_tv > 0 and volume is not None:
            from utils import total_variation_3d
            loss_tv = total_variation_3d(torch.abs(volume))
            loss_dict["loss_tv"] = loss_tv.item()

        # Total loss
        loss = loss_dc + lambda_tv * loss_tv
        loss_dict["loss_total"] = loss.item()

        return loss, loss_dict

    # ========================================================================
    # Adaptive Control: Densification & Pruning
    # ========================================================================

    def compute_center_gradients_norm(self) -> torch.Tensor:
        """计算 centers 梯度的 L2 范数。

        Returns:
            grad_norms: (M,) 每个 Gaussian center 的梯度范数
        """
        if self.centers.grad is None:
            return torch.zeros(self.n_gaussians, device=self.device)

        grad_norms = torch.norm(self.centers.grad, dim=1)
        return grad_norms

    def densify_and_prune(
        self,
        grad_threshold: float = 0.01,
        prune_rho_thresh: float = 0.001,
        max_gaussians: int = 50000,
        min_gaussians: int = 64,
        split_scale_factor: float = 0.7,
        split_delta_factor: float = 0.5,
    ) -> Dict[str, int]:
        """执行 densification 和 pruning。

        Densification（Long-axis splitting）:
        - 对梯度范数 > grad_threshold 的 Gaussian 进行分裂
        - 沿 scale 最大的轴方向分裂

        Pruning:
        - 删除 |rho| < prune_rho_thresh 的 Gaussian

        Args:
            grad_threshold: 分裂触发的梯度阈值
            prune_rho_thresh: 删除阈值
            max_gaussians: 最大 Gaussian 数量
            min_gaussians: pruning 后至少保留的 Gaussian 数量
            split_scale_factor: 分裂后 scale 缩小系数
            split_delta_factor: 分裂位移系数（delta = factor * sigma_k）

        Returns:
            stats: 包含 n_split, n_pruned, n_final 的字典
        """
        with torch.no_grad():
            stats = {"n_split": 0, "n_pruned": 0, "n_final": self.n_gaussians}

            if self.n_gaussians == 0:
                return stats

            # 1. Pruning: 删除低 density 的点
            rho_mag = torch.sqrt(self.rho_real**2 + self.rho_imag**2)
            keep_mask = rho_mag >= prune_rho_thresh
            n_pruned = (~keep_mask).sum().item()
            stats["n_pruned"] = n_pruned

            # 至少保留 min_gaussians 个（按 rho 能量最高）
            if keep_mask.sum() < min_gaussians:
                topk = torch.topk(rho_mag, k=min(self.n_gaussians, min_gaussians)).indices
                keep_mask[topk] = True

            if n_pruned > 0:
                self._prune_gaussians(keep_mask)

            # 如果被 prune 到空，直接返回，避免后续计算空 tensor 的 max
            if self.n_gaussians == 0:
                stats["n_final"] = 0
                return stats

            # 2. Densification: 分裂高梯度的点
            if self.n_gaussians < max_gaussians:
                grad_norms = self.compute_center_gradients_norm()

                # 归一化梯度（可选）
                if grad_norms.numel() == 0 or grad_norms.max() == 0:
                    stats["n_final"] = self.n_gaussians
                    return stats
                else:
                    grad_norms_normalized = grad_norms / (grad_norms.max() + 1e-8)

                split_mask = grad_norms_normalized > grad_threshold

                # 限制分裂数量
                n_can_split = min(
                    split_mask.sum().item(),
                    max_gaussians - self.n_gaussians,
                )

                if n_can_split > 0:
                    # 选择梯度最大的 n_can_split 个点
                    split_indices = torch.topk(
                        grad_norms * split_mask.float(),
                        k=n_can_split,
                    ).indices

                    self._split_gaussians(
                        split_indices,
                        split_scale_factor=split_scale_factor,
                        split_delta_factor=split_delta_factor,
                    )
                    stats["n_split"] = n_can_split

            stats["n_final"] = self.n_gaussians
            return stats

    def _prune_gaussians(self, keep_mask: torch.Tensor):
        """删除指定的 Gaussians。"""
        self.centers = nn.Parameter(self.centers[keep_mask])
        self.log_scales = nn.Parameter(self.log_scales[keep_mask])
        self.rho_real = nn.Parameter(self.rho_real[keep_mask])
        self.rho_imag = nn.Parameter(self.rho_imag[keep_mask])

        if self.use_rotation:
            self.rotations = nn.Parameter(self.rotations[keep_mask])
        else:
            self.rotations = self.rotations[keep_mask]

    def _split_gaussians(
        self,
        split_indices: torch.Tensor,
        split_scale_factor: float = 0.7,
        split_delta_factor: float = 0.5,
    ):
        """Long-axis splitting: 沿最大 scale 轴分裂。

        Args:
            split_indices: 要分裂的 Gaussian 索引
            split_scale_factor: 分裂后 scale 缩小系数
            split_delta_factor: 位移系数
        """
        n_split = split_indices.shape[0]
        if n_split == 0:
            return

        # 获取要分裂的 Gaussians
        centers_to_split = self.centers[split_indices]      # (n_split, 3)
        log_scales_to_split = self.log_scales[split_indices]  # (n_split, 3)
        rho_real_to_split = self.rho_real[split_indices]    # (n_split,)
        rho_imag_to_split = self.rho_imag[split_indices]    # (n_split,)

        scales_to_split = torch.exp(log_scales_to_split)    # (n_split, 3)

        # 找到最大 scale 的轴
        max_axis = torch.argmax(scales_to_split, dim=1)  # (n_split,)

        # 计算位移
        delta = split_delta_factor * scales_to_split[torch.arange(n_split), max_axis]  # (n_split,)

        # 构建方向向量（沿最大轴的单位向量）
        direction = torch.zeros(n_split, 3, device=self.device)
        direction[torch.arange(n_split), max_axis] = 1.0

        # 生成两个新的中心
        new_centers_plus = centers_to_split + delta.unsqueeze(1) * direction
        new_centers_minus = centers_to_split - delta.unsqueeze(1) * direction

        # 新的 log_scales（缩小）
        new_log_scales = log_scales_to_split + math.log(split_scale_factor)

        # 新的 rho（平分）
        new_rho_real = rho_real_to_split / 2
        new_rho_imag = rho_imag_to_split / 2

        # 合并新旧参数
        # 保留原有的（除了被分裂的）
        keep_mask = torch.ones(self.n_gaussians, dtype=torch.bool, device=self.device)
        keep_mask[split_indices] = False

        new_centers = torch.cat([
            self.centers[keep_mask],
            new_centers_plus,
            new_centers_minus,
        ], dim=0)

        new_log_scales = torch.cat([
            self.log_scales[keep_mask],
            new_log_scales,
            new_log_scales,
        ], dim=0)

        new_rho_real = torch.cat([
            self.rho_real[keep_mask],
            new_rho_real,
            new_rho_real,
        ], dim=0)

        new_rho_imag = torch.cat([
            self.rho_imag[keep_mask],
            new_rho_imag,
            new_rho_imag,
        ], dim=0)

        # 更新参数
        self.centers = nn.Parameter(new_centers)
        self.log_scales = nn.Parameter(new_log_scales)
        self.rho_real = nn.Parameter(new_rho_real)
        self.rho_imag = nn.Parameter(new_rho_imag)

        # Rotations
        if self.use_rotation:
            rotations_to_split = self.rotations[split_indices]
            new_rotations = torch.cat([
                self.rotations[keep_mask],
                rotations_to_split,
                rotations_to_split,
            ], dim=0)
            self.rotations = nn.Parameter(new_rotations)
        else:
            rotations_to_split = self.rotations[split_indices]
            new_rotations = torch.cat([
                self.rotations[keep_mask],
                rotations_to_split,
                rotations_to_split,
            ], dim=0)
            self.rotations = new_rotations

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取模型状态用于保存。"""
        state = {
            "centers": self.centers.data,
            "log_scales": self.log_scales.data,
            "rho_real": self.rho_real.data,
            "rho_imag": self.rho_imag.data,
            "rotations": self.rotations.data if self.use_rotation else self.rotations,
        }
        return state

    def load_state_dict_custom(self, state: Dict[str, torch.Tensor]):
        """从保存的状态恢复模型。"""
        n_gaussians = state["centers"].shape[0]

        self.centers = nn.Parameter(state["centers"].to(self.device))
        self.log_scales = nn.Parameter(state["log_scales"].to(self.device))
        self.rho_real = nn.Parameter(state["rho_real"].to(self.device))
        self.rho_imag = nn.Parameter(state["rho_imag"].to(self.device))

        if self.use_rotation:
            self.rotations = nn.Parameter(state["rotations"].to(self.device))
        else:
            self.rotations = state["rotations"].to(self.device)


if __name__ == "__main__":
    # 简单测试
    print("=== Model Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    volume_shape = (32, 64, 64)

    # 创建模型
    model = GaussianMRIModel(
        volume_shape=volume_shape,
        n_gaussians=100,
        device=device,
    )
    print(f"Initial Gaussians: {model.n_gaussians}")

    # 创建 mock 初始图像
    image_init = torch.randn(*volume_shape, dtype=torch.complex64, device=device)

    # 从图像初始化
    model.initialize_from_image(
        image_init,
        n_gaussians=500,
        percentile_thresh=10.0,
    )
    print(f"After init: {model.n_gaussians} Gaussians")

    # 体素化
    volume = model.voxelize()
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

    # Forward
    mask = torch.ones(*volume_shape, device=device)
    recon, kspace_pred = model.forward(mask)
    print(f"Recon shape: {recon.shape}, K-space shape: {kspace_pred.shape}")

    # Loss
    kspace_target = torch.randn_like(kspace_pred)
    loss, loss_dict = model.compute_loss(kspace_pred, kspace_target, mask, volume, lambda_tv=0.001)
    print(f"Loss: {loss.item():.6f}, Dict: {loss_dict}")

    # 反向传播（测试梯度）
    loss.backward()
    print(f"Centers grad norm: {model.centers.grad.norm().item():.6f}")

    # Densify/Prune
    stats = model.densify_and_prune(grad_threshold=0.01, prune_rho_thresh=0.0001)
    print(f"Densify stats: {stats}")

    print("Model test passed!")
