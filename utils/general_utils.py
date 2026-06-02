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
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))




def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # Normalize the batch of direction vectors
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    # Create a batch of arbitrary vectors that are not collinear with the direction vectors
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(direction_vectors.device).expand(direction_vectors.shape[0], -1).clone()
    is_collinear = torch.all(torch.abs(direction_vectors - v1) < 1e-5, dim=-1)
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(direction_vectors.device)

    # Calculate the first orthogonal vectors
    v1 = torch.cross(direction_vectors, v1)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True))
    # Calculate the second orthogonal vectors by taking the cross product
    v2 = torch.cross(direction_vectors, v1)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True))
    # Create the batch of rotation matrices with the direction vectors as the last columns
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices

# from kornia.geometry import conversions
# def normal_to_rotation(normals):
#     rotations = create_rotation_matrix_from_direction_vector_batch(normals)
#     rotations = conversions.rotation_matrix_to_quaternion(rotations,eps=1e-5, order=conversions.QuaternionCoeffOrder.WXYZ)
#     return rotations


def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img

def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:,0,:] # normalized by defaut

    return x_axis

def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True) # (N, 1)
    non_flip = dotprod>=0 # (N, 1)
    normal_flipped = normal*torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip

import os
import torchvision
import cv2
def save_image(iteration, model_path, viewpoint_cam, render_pkg, ref_render_pkg, pbr_rgb):          
    surf_depth = render_pkg["surf_depth"] / render_pkg["surf_depth"].max()
    
    gt_image = viewpoint_cam.original_image.cuda()

    os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)
    row0 = torch.cat([gt_image[:3], pbr_rgb, render_pkg["rend_alpha"].repeat(3,1,1), render_pkg["rend_roughness"].repeat(3,1,1)], dim=2)
    row1 = torch.cat([render_pkg["output_diff"], ref_render_pkg["output_spec"], ((render_pkg['rend_normal']+1)/2), (render_pkg['rend_normal']+1)/2], dim=2)
    #row2 = torch.cat([ref_render_pkg["rend_alpha"].repeat(3,1,1), ref_render_pkg["vis_map"].repeat(3,1,1), ((ref_render_pkg["rend_normal"]+1)/2), (ref_render_pkg["surf_normal"]+1)/2], dim=2)
    image_to_show = torch.cat([row0, row1], dim=1)
    
    labels = [
        ["GT", "Render", "Alpha", "Roughness"],
        ["Diffuse", "Specular", "Normal", "Depth Normal"],
        ["Local Alpha", "Indi_W", "Local Normal", "Local Depth Normal"]
    ]

    # 1. 将 Tensor 转换为 Numpy 格式 (H, W, C), 范围 [0, 255], 类型 uint8
    grid_numpy = image_to_show.detach().cpu().permute(1, 2, 0).numpy()
    grid_numpy = (grid_numpy * 255).clip(0, 255).astype(np.uint8)

    # 2. 如果是 RGB，转为 BGR (因为 OpenCV 使用 BGR 顺序保存图片)
    # 假设 PyTorch Tensor 是 RGB 顺序
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_RGB2BGR)

    # 3. 计算单个子图的宽高
    total_h, total_w, _ = grid_numpy.shape
    rows = 2
    cols = 4
    h_step = total_h // rows
    w_step = total_w // cols

    # 4. 循环在每个格子上画字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # 字体大小，可根据图片分辨率调整
    thickness = 3     # 线条粗细
    color = (0, 255, 255) # 红色 (B, G, R)

    for r in range(rows):
        for c in range(cols):
            text = labels[r][c]
            
            # 计算文字坐标 (左上角 + 偏移量)
            x_pos = c * w_step + 20  # 距左边 10 像素
            y_pos = r * h_step + 60  # 距顶边 60 像素
            
            # 可选：画黑色描边以增强对比度 (背景太亮时也能看清)
            cv2.putText(grid_numpy, text, (x_pos, y_pos), font, font_scale, (0,0,0), thickness + 2)
            # 画彩色文字
            cv2.putText(grid_numpy, text, (x_pos, y_pos), font, font_scale, color, thickness)

    # 5. 保存图片 (使用 cv2.imwrite 替代 torchvision.save_image)
    save_path = os.path.join(model_path, "debug", "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg")
    cv2.imwrite(save_path, grid_numpy)
                
def save_image_real(iteration, model_path, viewpoint_cam, render_pkg, ref_render_pkg, pbr_rgb):         
    surf_depth = render_pkg["surf_depth"] / render_pkg["surf_depth"].max()
                
    gt_image = viewpoint_cam.original_image.cuda()

    os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)
    row0 = torch.cat([gt_image[:3], pbr_rgb, render_pkg["rend_alpha"].repeat(3,1,1), render_pkg["rend_roughness"].repeat(3,1,1)], dim=2)
    row1 = torch.cat([render_pkg["output_diff"], ref_render_pkg["output_spec"], ((render_pkg['rend_normal']+1)/2), (render_pkg['rend_normal']+1)/2], dim=2)
    row2 = torch.cat([render_pkg["render"], render_pkg["out_w"].repeat(3,1,1), render_pkg["ref_w"].repeat(3,1,1), surf_depth.repeat(3,1,1)], dim=2)
    #row3 = torch.cat([ref_render_pkg["rend_alpha"].repeat(3,1,1), ref_render_pkg["vis_map"].repeat(3,1,1), ((ref_render_pkg["rend_normal"]+1)/2), (ref_render_pkg["surf_normal"]+1)/2], dim=2)
    image_to_show = torch.cat([row0, row1, row2], dim=1)
    
    labels = [
        ["GT", "Render", "Alpha", "Roughness"],
        ["Diffuse", "Specular", "Normal", "Depth Normal"],
        ["SH Render", "Out_W", "Ref_W", "Depth"],
        ["Local Alpha", "Indi_W", "Local Normal", "Local Depth Normal"]
    ]

    # 1. 将 Tensor 转换为 Numpy 格式 (H, W, C), 范围 [0, 255], 类型 uint8
    grid_numpy = image_to_show.detach().cpu().permute(1, 2, 0).numpy()
    grid_numpy = (grid_numpy * 255).clip(0, 255).astype(np.uint8)

    # 2. 如果是 RGB，转为 BGR (因为 OpenCV 使用 BGR 顺序保存图片)
    # 假设 PyTorch Tensor 是 RGB 顺序
    grid_numpy = cv2.cvtColor(grid_numpy, cv2.COLOR_RGB2BGR)

    # 3. 计算单个子图的宽高
    total_h, total_w, _ = grid_numpy.shape
    rows = 3
    cols = 4
    h_step = total_h // rows
    w_step = total_w // cols

    # 4. 循环在每个格子上画字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5  # 字体大小，可根据图片分辨率调整
    thickness = 3     # 线条粗细
    color = (0, 255, 255) # (B, G, R)

    for r in range(rows):
        for c in range(cols):
            text = labels[r][c]
            
            # 计算文字坐标 (左上角 + 偏移量)
            x_pos = c * w_step + 20  # 距左边 10 像素
            y_pos = r * h_step + 60  # 距顶边 60 像素
            
            # 可选：画黑色描边以增强对比度 (背景太亮时也能看清)
            cv2.putText(grid_numpy, text, (x_pos, y_pos), font, font_scale, (0,0,0), thickness + 2)
            # 画彩色文字
            cv2.putText(grid_numpy, text, (x_pos, y_pos), font, font_scale, color, thickness)

    # 5. 保存图片 (使用 cv2.imwrite 替代 torchvision.save_image)
    save_path = os.path.join(model_path, "debug", "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg")
    cv2.imwrite(save_path, grid_numpy)