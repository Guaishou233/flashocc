# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss
import numpy as np


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight

    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.
    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    num_classes = pred.size(1)
    target = F.one_hot(target, num_classes=num_classes + 1)
    target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight

    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            elif weight.size(0) == loss.size(1):
                # weight is of shape (num_classes, ), need to broadcast to (num_priors, num_classes)
                weight = weight.view(1, -1).expand_as(loss)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                if weight.numel() == loss.numel():
                    weight = weight.view(loss.size(0), -1)
                elif weight.numel() == loss.size(1):
                    # weight is (num_classes, ), broadcast to (num_priors, num_classes)
                    weight = weight.view(1, -1).expand_as(loss)
                else:
                    raise ValueError(f"weight shape {weight.shape} cannot be broadcast to loss shape {loss.shape}. "
                                   f"Expected weight to have shape (num_priors,), (num_classes,), or (num_priors, num_classes).")
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class CustomFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=100.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(CustomFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated
        H, W = 200, 200

        xy, yx = torch.meshgrid([torch.arange(H) - H / 2, torch.arange(W) - W / 2])
        c = torch.stack([xy, yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c / c_max + 1).cuda()

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                ignore_index=255,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        # 处理target的形状：如果是6维(B, 1, 1, Dx, Dy, Dz)，先reshape成4维(B, Dx, Dy, Dz)
        if target.dim() == 6:
            # (B, 1, 1, Dx, Dy, Dz) -> (B, Dx, Dy, Dz)
            target = target.squeeze(1).squeeze(1)
        elif target.dim() == 5:
            # 如果已经是5维，可能需要处理
            # (B, C, Dx, Dy, Dz) -> (B, Dx, Dy, Dz) 取第一个通道或者reshape
            if target.size(1) == 1:
                target = target.squeeze(1)
        
        # 现在target应该是(B, Dx, Dy, Dz)或(B, H, W, D)
        # 代码期望的是(B, H, W, D)格式，其中D是最后一个维度
        if target.dim() == 4:
            B, H, W, D = target.shape
        elif target.dim() == 3:
            # 如果是3维，可能是(B, H, W)或(B, H*W, D)
            # 根据实际情况处理
            B = target.shape[0]
            if target.shape[-1] > target.shape[1]:
                # 可能是(B, H, D)格式，需要reshape
                H, W, D = target.shape[1:]
            else:
                # 可能是(B, H, W)格式，需要添加D维度
                H, W = target.shape[1:]
                D = 1
                target = target.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected target shape: {target.shape}")

        # 动态创建c矩阵，如果H和W与self.c的尺寸不同
        if H == self.c.shape[0] and W == self.c.shape[1]:
            c = self.c[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)
        else:
            # 根据实际的H和W动态创建c，与初始化代码保持一致
            xy, yx = torch.meshgrid([torch.arange(H, device=target.device, dtype=target.dtype) - H / 2, 
                                     torch.arange(W, device=target.device, dtype=target.dtype) - W / 2])
            c = torch.stack([xy, yx], 2)
            c = torch.norm(c, 2, -1)
            c_max = c.max()
            if c_max > 0:
                c = (c / c_max + 1)
            else:
                c = torch.ones_like(c)
            c = c[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)

        visible_mask = (target != ignore_index).reshape(-1).nonzero().squeeze(-1)
        
        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]
        
        # 处理weight的形状：确保weight_mask的形状与pred和target匹配
        if weight is not None:
            num_visible = visible_mask.size(0)
            if weight.dim() == 1:
                # weight是(num_classes,)，需要广播到(num_visible, num_classes)
                if weight.size(0) == num_classes:
                    # weight是(num_classes,)，广播到(num_visible, num_classes)
                    # weight_mask = weight[None, :] * c[visible_mask, None]  # (num_visible, num_classes)
                    # 或者直接使用weight，让loss函数处理
                    weight_mask = weight[None, :].expand(num_visible, num_classes) * c[visible_mask, None]
                else:
                    # weight可能是其他形状，尝试reshape
                    weight_mask = weight[visible_mask] if weight.size(0) >= num_visible else weight
            elif weight.dim() == 2:
                # weight已经是(B*H*W*D, num_classes)或(B*H*W*D, 1)
                if weight.size(0) == B * H * W * D:
                    # weight是(B*H*W*D, num_classes)或(B*H*W*D, 1)
                    weight_mask = weight[visible_mask]  # (num_visible, num_classes)或(num_visible, 1)
                    # 如果需要，将其与c相乘
                    if weight_mask.size(1) == 1:
                        weight_mask = weight_mask * c[visible_mask, None]  # (num_visible, 1)
                        # 如果需要扩展到(num_visible, num_classes)
                        weight_mask = weight_mask.expand(num_visible, num_classes)
                    else:
                        weight_mask = weight_mask * c[visible_mask, None]  # (num_visible, num_classes)
                else:
                    # weight的形状不匹配，尝试reshape
                    # 如果weight的总元素数等于B*H*W*D*num_classes，尝试reshape
                    if weight.numel() == B * H * W * D * num_classes:
                        weight_mask = weight.view(B * H * W * D, num_classes)[visible_mask]
                    elif weight.numel() == B * H * W * D:
                        weight_mask = weight.view(B * H * W * D, 1)[visible_mask]
                        weight_mask = weight_mask * c[visible_mask, None]
                        weight_mask = weight_mask.expand(num_visible, num_classes)
                    else:
                        # 如果都不匹配，尝试直接使用
                        weight_mask = weight[visible_mask] if weight.size(0) >= num_visible else weight
            else:
                # weight是其他维度，尝试reshape
                weight_mask = weight.view(-1, weight.size(-1))[visible_mask] if weight.numel() >= num_visible * num_classes else weight
        else:
            weight_mask = None

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target.to(torch.long),
                weight_mask,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
