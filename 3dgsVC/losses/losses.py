"""
Loss Functions for 3DGSMR

论文公式(2): X* = argmin ||A(X) - b||^2 + λ·R(X)

包含:
- k-space域loss (数据一致性)
- image域loss
- TV正则化 (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class KSpaceLoss(nn.Module):
    """
    K-space域损失函数
    
    论文公式(2)的数据一致性项: ||A(X) - b||^2
    
    只在采样位置计算loss (使用mask)
    """
    
    def __init__(self, loss_type: str = "l1"):
        """
        Args:
            loss_type: "l1" 或 "l2"
        """
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算k-space loss
        
        Args:
            kspace_pred: 预测的k-space (D, H, W), complex
            kspace_target: 目标k-space (D, H, W), complex
            mask: 采样mask (D, H, W)
            
        Returns:
            loss值
        """
        # 只在采样位置计算
        kspace_pred_masked = kspace_pred * mask
        kspace_target_masked = kspace_target * mask
        
        # 计算差异
        diff = kspace_pred_masked - kspace_target_masked
        
        if self.loss_type == "l1":
            # L1 loss on complex values (use absolute value)
            loss = torch.abs(diff).mean()
        else:  # l2
            # L2 loss
            loss = (torch.abs(diff) ** 2).mean()
            
        return loss


class ImageLoss(nn.Module):
    """
    图像域损失函数
    
    在图像域计算重建质量
    """
    
    def __init__(self, loss_type: str = "l1"):
        """
        Args:
            loss_type: "l1" 或 "l2"
        """
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        image_pred: torch.Tensor,
        image_target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算图像域loss
        
        Args:
            image_pred: 预测图像 (D, H, W), complex
            image_target: 目标图像 (D, H, W), complex
            weight_map: 可选的权重图
            
        Returns:
            loss值
        """
        # 计算幅度差异
        pred_mag = torch.abs(image_pred)
        target_mag = torch.abs(image_target)
        
        diff = pred_mag - target_mag
        
        if weight_map is not None:
            diff = diff * weight_map
        
        if self.loss_type == "l1":
            loss = torch.abs(diff).mean()
        else:  # l2
            loss = (diff ** 2).mean()
            
        return loss


class ComplexImageLoss(nn.Module):
    """
    复数图像域损失函数
    
    同时考虑实部和虚部
    """
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(
        self,
        image_pred: torch.Tensor,
        image_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_pred: 预测图像 (D, H, W), complex
            image_target: 目标图像 (D, H, W), complex
        """
        # 分别计算实部和虚部的loss
        real_diff = image_pred.real - image_target.real
        imag_diff = image_pred.imag - image_target.imag
        
        if self.loss_type == "l1":
            loss = torch.abs(real_diff).mean() + torch.abs(imag_diff).mean()
        else:
            loss = (real_diff ** 2).mean() + (imag_diff ** 2).mean()
            
        return loss


class TVLoss(nn.Module):
    """
    Total Variation正则化
    
    论文Section V指出TV在MRI重建中效果有限,
    但对于某些情况仍可使用
    
    R(X) = ||∇X||_1
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算3D Total Variation
        
        Args:
            image: 输入图像 (D, H, W), 可以是complex
            
        Returns:
            TV loss值
        """
        # 使用幅度图
        if image.is_complex():
            image = torch.abs(image)
            
        # 计算三个方向的差分
        diff_d = torch.abs(image[1:, :, :] - image[:-1, :, :])
        diff_h = torch.abs(image[:, 1:, :] - image[:, :-1, :])
        diff_w = torch.abs(image[:, :, 1:] - image[:, :, :-1])
        
        # 求和
        tv_loss = diff_d.mean() + diff_h.mean() + diff_w.mean()
        
        return tv_loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    总loss = kspace_weight * kspace_loss + image_weight * image_loss + tv_weight * tv_loss
    """
    
    def __init__(
        self,
        kspace_weight: float = 1.0,
        image_weight: float = 0.1,
        tv_weight: float = 0.0,
        loss_type: str = "l1"
    ):
        """
        Args:
            kspace_weight: k-space loss权重
            image_weight: image loss权重
            tv_weight: TV loss权重
            loss_type: loss类型 ("l1" or "l2")
        """
        super().__init__()
        
        self.kspace_weight = kspace_weight
        self.image_weight = image_weight
        self.tv_weight = tv_weight
        
        self.kspace_loss = KSpaceLoss(loss_type)
        self.image_loss = ImageLoss(loss_type)
        self.tv_loss = TVLoss()
        
    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_target: torch.Tensor,
        mask: torch.Tensor,
        image_pred: Optional[torch.Tensor] = None,
        image_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合loss
        
        Args:
            kspace_pred: 预测k-space
            kspace_target: 目标k-space
            mask: 采样mask
            image_pred: 预测图像 (可选，用于image loss和TV)
            image_target: 目标图像 (可选)
            
        Returns:
            包含各项loss的字典
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=kspace_pred.device)
        
        # K-space loss (数据一致性)
        kspace_l = self.kspace_loss(kspace_pred, kspace_target, mask)
        losses['kspace_loss'] = kspace_l
        total_loss = total_loss + self.kspace_weight * kspace_l
        
        # Image loss
        if self.image_weight > 0 and image_pred is not None and image_target is not None:
            image_l = self.image_loss(image_pred, image_target)
            losses['image_loss'] = image_l
            total_loss = total_loss + self.image_weight * image_l
        
        # TV loss
        if self.tv_weight > 0 and image_pred is not None:
            tv_l = self.tv_loss(image_pred)
            losses['tv_loss'] = tv_l
            total_loss = total_loss + self.tv_weight * tv_l
            
        losses['total_loss'] = total_loss
        
        return losses


class DataConsistencyLayer(nn.Module):
    """
    数据一致性层
    
    在采样位置替换为观测值
    对应论文中的data consistency constraint
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        kspace_pred: torch.Tensor,
        kspace_measured: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        应用数据一致性
        
        Args:
            kspace_pred: 预测的k-space
            kspace_measured: 测量的k-space
            mask: 采样mask
            
        Returns:
            数据一致性处理后的k-space
        """
        # 在采样位置使用测量值，未采样位置使用预测值
        kspace_dc = kspace_pred * (1 - mask) + kspace_measured * mask
        return kspace_dc
