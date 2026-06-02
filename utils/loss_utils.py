#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def entropy_loss(alpha):
    loss = -alpha * torch.log(alpha + 1e-10) - (1 - alpha) * torch.log(1 - alpha + 1e-10)
    loss = torch.mean(loss)
    return loss


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input + 1e-10) + (1 - target) * torch.log(1 - input + 1e-10)).mean()


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def depth_prior_loss(rendered_depth: torch.Tensor, prior_depth: torch.Tensor, prior_depth_conf: torch.Tensor, mask: torch.Tensor = None):
    """
    计算基于最小二乘法对齐的深度先验损失。

    参数:
        rendered_depth (torch.Tensor): 从3DGS渲染得到的深度图，形状为 [1, H, W] 或 [H, W]。(通常是米或场景单位)。
        prior_depth (torch.Tensor): 深度先验图，例如VGGt计算的深度，形状为 [H, W]。(通常是任意尺度)。
        prior_depth_conf (torch.Tensor): VGGt生成的置信度图，形状为 [H, W]，归一化到 [0, 1] 之间，越高表示越可信。
        mask (torch.Tensor, optional): 一个布尔掩码，指示哪些像素是有效的（例如，非零深度）。形状为 [H, W] 或 [1, H, W]。
                                            如果为None，则会自动创建一个掩码来排除零值。

    返回:
        torch.Tensor: 计算出的深度先验损失（标量）。
    """

    # 假设 rendered_depth, prior_depth, prior_depth_conf, mask 
    # 在传入时已经处理好了类型、设备和有效性检查。

    # --- 1. 提取有效像素的深度值 ---
    D_model_flat = rendered_depth[mask] # 扁平化的渲染深度
    D_prior_flat = prior_depth[mask] # 扁平化的先验深度
    conf_flat = prior_depth_conf[mask] # 扁平化的置信度

    # 如果没有足够的有效像素来执行最小二乘法，返回0损失
    # （尽管你提到已在外部检查，但在这里再加一个防御性检查也无妨，不会引入额外内存）
    if D_model_flat.numel() < 2:
        return torch.tensor(0.0, device=rendered_depth.device)

    # --- 2. 构建用于最小二乘法求解的矩阵 A 和向量 B ---
    # 我们要解的是 D_model_flat * omega + b = D_prior_flat
    # 对应 Ax = B，其中 x = [omega, b]^T
    
    # 计算权重的平方根用于加权最小二乘
    sqrt_weights = torch.sqrt(conf_flat)
    
    # A 矩阵的每一行是 [sqrt(w_i) * D_model_flat[i], sqrt(w_i)]
    A = torch.stack((sqrt_weights * D_model_flat, sqrt_weights), dim=1) # shape: [num_valid_pixels, 2]
    
    # B 向量是 sqrt(w_i) * D_prior_flat
    B = sqrt_weights * D_prior_flat # shape: [num_valid_pixels]

    # --- 4. 求解 omega 和 b ---
    # 使用 torch.linalg.lstsq 来求解最小二乘问题，它对数值稳定性更鲁棒
    # 返回值包括 solution (即 [omega, b]), residuals, rank, singular_values
    
    # FIX 1: 正确解包 torch.linalg.lstsq 的所有四个返回值
    # FIX 2: 将h从计算图中分离 (detach())
    h, _, _, _ = torch.linalg.lstsq(A, B) 
    h = h.detach() # 关键：将解出的参数从计算图中分离，避免lstsq的反向传播
    
    omega, b = h[0], h[1]

    # --- 5. 计算对齐后的深度 ---
    # D_aligned_flat 仍然会依赖于 D_model_flat (即 rendered_depth)，所以梯度会正常流回
    D_aligned_flat = omega * D_model_flat + b

    # --- 6. 计算深度先验损失 ---
    # 使用加权 L2 损失，置信度高的像素权重更大
    weighted_loss = conf_flat * (D_aligned_flat - D_prior_flat)**2
    loss = torch.mean(weighted_loss)

    return loss