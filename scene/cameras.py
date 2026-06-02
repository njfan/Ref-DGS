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

import torch, torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
import torchvision.transforms as transforms

class Camera(nn.Module):
    def __init__(self, 
                    colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                    image_path, image_name, uid,
                    trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",              
                    rays_o=None, rays_d=None,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = torch.tensor(R, dtype=torch.float32, device='cuda')
        self.T = torch.tensor(T, dtype=torch.float32, device='cuda')
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.nearest_id = []
        self.nearest_names = []

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = (gt_alpha_mask).to(self.data_device)
        else:
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.original_image *= self.gt_alpha_mask
            
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        self.rays_o = rays_o
        self.rays_d = rays_d
        
        KEYWORDS = ["ref_real", "GlossyReal"]

        if any(k in image_path for k in KEYWORDS):

            transform = transforms.Compose([
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor(),
            ])
            if "ref_real" in image_path:
                depth_path=os.path.join("priors/Ref-NeRF/ref_real", Path(image_path).parent.parent.name, "depth", Path(image_path).name[:-3]+"pth")
                normal_path=os.path.join("priors/Ref-NeRF/ref_real", Path(image_path).parent.parent.name, "normal", Path(image_path).name)
            elif "GlossyReal" in image_path:
                depth_path=os.path.join("priors/Glossy/GlossyReal", Path(image_path).parent.parent.name, "depth", Path(image_path).name[:-3]+"pth")
                normal_path=os.path.join("priors/Glossy/GlossyReal", Path(image_path).parent.parent.name, "normal", Path(image_path).name)
            normal = transform(Image.open(normal_path)).cuda()
            normal = normal[[2, 1, 0], :, :] # bgr->rgb
            self.vggt_normal = (torch.nn.functional.normalize(normal * 2.0 - 1.0, dim=0).permute(1, 2, 0) @ self.world_view_transform[:3,:3].T).permute(2, 0, 1)  # c2w
            vggt_depth = torch.load(depth_path) # [H, W]
            self.vggt_depth = torch.nn.functional.interpolate(
                vggt_depth[None, None, ...],     # [1,1,H,W]
                size=(self.image_height, self.image_width),
                mode="nearest"
            ).squeeze(0).to(self.data_device)         # [1, H, W]
            
            self.vggt_depth_conf = torch.ones_like(self.vggt_depth)            
            
        else:
            if "refnerf" in image_path:
                depth_path=os.path.join("priors/Ref-NeRF/refnerf", Path(image_path).parent.parent.name, Path(image_path).parent.name, "depth", Path(image_path).name[:-3]+"pth")
                normal_path=os.path.join("priors/Ref-NeRF/refnerf", Path(image_path).parent.parent.name, Path(image_path).parent.name, "normal", Path(image_path).name)
            elif "GlossySynthetic" in image_path:
                depth_path=os.path.join("priors/Glossy/GlossySynthetic", Path(image_path).parent.parent.name, "depth", Path(image_path).name[:-3]+"pth")
                normal_path=os.path.join("priors/Glossy/GlossySynthetic", Path(image_path).parent.parent.name, "normal", Path(image_path).name)

            data = torch.load(depth_path)
            
            self.vggt_depth = F.interpolate(data["depth_map"].unsqueeze(0).unsqueeze(0), size=(self.image_height, self.image_width), 
                                        mode="bilinear", align_corners=False).squeeze(0).to(self.data_device)  # no_mask  []

            vggt_depth_conf  = F.interpolate(data["depth_conf"].unsqueeze(0).unsqueeze(0), size=(self.image_height, self.image_width),
                                        mode="nearest").squeeze(0).to(self.data_device)
            
            conf_log = torch.log(vggt_depth_conf + 1.0)

            conf_norm = (conf_log - conf_log.min()) / (conf_log.max() - conf_log.min() + 1e-8)
            conf_norm[~(self.gt_alpha_mask>0.5)] = 0.0
            
            tau = 1
            self.vggt_depth_conf = conf_norm ** tau
            self.vggt_normal = None
            
            self.vggt_normal = (((torch.from_numpy(np.array(Image.open(normal_path))) / 255.0)*2-1).cuda() @ (self.world_view_transform[:3,:3].T)).permute(2, 0, 1)[:3, ]  # c2w
        # # save depth to normal
        # from utils.point_utils import depth_to_normal
        # out_path = os.path.join(Path(image_path).parent, "vggt_normal", Path(image_path).stem + ".png")  # 局部法向
        # os.makedirs(os.path.join(Path(image_path).parent, "vggt_normal"), exist_ok=True)
        # # vggt_normal_local = depth_to_normal(self, self.vggt_depth * (self.gt_alpha_mask>0.5))
        # # vggt_normal_global = vggt_normal_local @ self.world_view_transform[:3,:3].T
        # vggt_normal_global = depth_to_normal(self, self.vggt_depth * (self.gt_alpha_mask>0.5))
        # # conf_normal_no_mask = vggt_normal_local *conf_norm_raw.unsqueeze(-1)
        # # conf_normal_mask = vggt_normal_local *self.vggt_depth_conf.unsqueeze(-1)
        # # row0 = np.concatenate([fun(vggt_normal_local), fun(vggt_normal_global)], axis=1)
        # # row1 = np.concatenate([fun(conf_normal_no_mask), fun(conf_normal_mask)], axis=1)            
        # # image_to_show = np.concatenate([row0, row1], axis=0)
        # # cv2.imwrite(out_path, image_to_show)
        # if torch.isnan(vggt_normal_global).any() or torch.isinf(vggt_normal_global).any():
        #     print(f"[Warning] NaN/Inf detected in normal map for image: {out_path}")
        #     nan_count = torch.isnan(vggt_normal_global).sum().item()
        #     inf_count = torch.isinf(vggt_normal_global).sum().item()
        #     print(f"    NaN count: {nan_count}, Inf count: {inf_count}")
        # torchvision.utils.save_image(((vggt_normal_global+1)*0.5).permute(2, 0, 1), out_path)
    
        
    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

