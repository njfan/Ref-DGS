#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os, time
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
import torchvision
from pathlib import Path

from utils.color_utils import get_final_color
from utils.image_utils import psnr, mae
from utils.loss_utils import ssim
from lpipsPyTorch import lpips

from PIL import Image

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic=o3d.camera.PinholeCameraIntrinsic(width=viewpoint_cam.image_width, 
                    height=viewpoint_cam.image_height, 
                    cx = viewpoint_cam.image_width/2,
                    cy = viewpoint_cam.image_height/2,
                    fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.)),
                    fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.)))

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, ref_gaussians, render, render_ref, pipe, bg_color=None, ENV_CENTER=None, ENV_RADIUS=None, XYZ=[0,1,2], dataset=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.ref_gaussians = ref_gaussians
        self.dataset = dataset
        
        if ENV_CENTER==None:
            self.render = partial(render, pipe=pipe, bg_color=background)
        else:
            self.render = partial(render, pipe=pipe, bg_color=background, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ)
        self.render_ref = partial(render_ref, pipe=pipe, bg_color=background, XYZ=XYZ)
        
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        # self.ref_normals = []
        self.depth_normals = []
        self.viewpoint_stack = []
        self.diffuse = []
        self.specular = []
        self.roughness = []
        self.ref_w = []
        self.out_w = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        ssims = []
        psnrs = []
        lpipss = []
        render_times = []
        maes = []
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            gt_image = viewpoint_cam.original_image.cuda()
            bg = torch.ones(3, device="cuda")
            gt = gt_image[:3,...] * viewpoint_cam.gt_alpha_mask + (1-viewpoint_cam.gt_alpha_mask)
            
            t1 = time.time()
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            ref_render_pkg = self.render_ref(viewpoint_cam, self.ref_gaussians, render_pkg=render_pkg)
            render_time = time.time() - t1
            
            rgb = get_final_color(render_pkg, ref_render_pkg, bg)
        
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            # ref_normal = torch.nn.functional.normalize(ref_render_pkg['surf_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            # self.ref_normals.append(ref_normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            self.diffuse.append(render_pkg["output_diff"].cpu())
            self.specular.append(ref_render_pkg["output_spec"].cpu())
            self.roughness.append(render_pkg["rend_roughness"].cpu())
            if "ref_w" in render_pkg:
                self.ref_w.append(render_pkg["ref_w"].cpu())
                self.out_w.append(render_pkg["out_w"].cpu())
            ssims.append(ssim(rgb, gt).item())
            psnrs.append(psnr(rgb , gt).mean().item())
            lpipss.append(lpips(rgb, gt, net_type='vgg').item())
            render_times.append(render_time)
            
            if self.dataset != None:
                img_path = Path(viewpoint_cam.image_path)
                normal_mae = (render_pkg['rend_normal']*alpha+(1-alpha)).detach().cpu().numpy()
                if self.dataset == "shiny":
                    gt_normal_path = os.path.join(img_path.parent, viewpoint_cam.image_name + "_normal.png")
                    gt_normal = Image.open(gt_normal_path)
                    gt_normal_mae = (np.array(gt_normal)[..., :3] / 255).transpose(2, 0, 1)  # [H, W, 3] in range [0, 1]
                    gt_normal_mae = (gt_normal_mae - 0.5) * 2.0
                elif self.dataset == "glossy":
                    from utils.point_utils import depth_to_normal
                    gt_depth_path = os.path.join(img_path.parent.parent, "depth", viewpoint_cam.image_name+"-depth.png")
                    gt_depth = torch.from_numpy(np.array(Image.open(gt_depth_path))/255).unsqueeze(0).cuda().float()
                    gt_normal = depth_to_normal(viewpoint_cam, gt_depth).permute(2, 0, 1)
                    gt_normal_mae = (gt_normal*viewpoint_cam.gt_alpha_mask + (1-viewpoint_cam.gt_alpha_mask)).cpu().numpy()
                    
                    border_mask = np.zeros((viewpoint_cam.image_height, viewpoint_cam.image_width), dtype=bool)
                    border_mask[0, :]  = True
                    border_mask[-1, :] = True
                    border_mask[:, 0]  = True
                    border_mask[:, -1] = True

                    gt_normal_mae[:, border_mask] = np.array([1.0, 1.0, 1.0])[:, None]
                    normal_mae[:, border_mask] = np.array([1.0, 1.0, 1.0])[:, None]

                normal_mae = normal_mae / np.linalg.norm(normal_mae, axis=0, ord=2, keepdims=True)
                gt_normal_mae = gt_normal_mae / np.linalg.norm(gt_normal_mae, axis=0, keepdims=True)
                maes.append(mae(gt_normal_mae, normal_mae))
            
            
        self.ssim_v = np.array(ssims).mean()
        self.psnr_v = np.array(psnrs).mean()
        self.lpip_v = np.array(lpipss).mean()
        self.fps = 1.0 / np.array(render_times).mean()
        if self.dataset != None:
            self.mae_v = np.array(maes).mean()
            
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.diffuse = torch.stack(self.diffuse, dim=0)
        self.specular = torch.stack(self.specular, dim=0)
        self.roughness = torch.stack(self.roughness, dim=0)
        if "ref_w" in self.ref_w:
            self.ref_w = torch.stack(self.ref_w, dim=0)
            self.out_w = torch.stack(self.out_w, dim=0)
        
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            
            # if we have mask provided, use it
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, normalmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sampled_normal = torch.nn.functional.grid_sample(normalmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, sampled_normal, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, normal, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    normalmap = self.depth_normals[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        diffuse_path = os.path.join(path, "diffuse")
        specualr_path = os.path.join(path, "specular")
        roughness_path = os.path.join(path, "roughness")
        
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(diffuse_path, exist_ok=True)
        os.makedirs(specualr_path, exist_ok=True)
        os.makedirs(roughness_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(os.path.join(vis_path, 'depth'), exist_ok=True)
        os.makedirs(os.path.join(vis_path, 'normal'), exist_ok=True)
        os.makedirs(os.path.join(vis_path, 'ref_normal'), exist_ok=True)
        os.makedirs(os.path.join(vis_path, 'depth_normal'), exist_ok=True)
        os.makedirs(os.path.join(vis_path, 'torch_normal'), exist_ok=True)
        os.makedirs(os.path.join(vis_path, 'torch_depth_normal'), exist_ok=True)
        if self.ref_w:
            os.makedirs(os.path.join(vis_path, 'ref_w'), exist_ok=True)
            os.makedirs(os.path.join(vis_path, 'out_w'), exist_ok=True)
        
        if self.psnr_v:
            print('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(self.psnr_v, self.ssim_v, self.lpip_v, self.fps))
            dump_path = os.path.join(path, 'metric.txt')
            with open(dump_path, 'w') as f:
                f.write('psnr:{}, ssim:{}, lpips:{}, fps:{}'.format(self.psnr_v, self.ssim_v, self.lpip_v, self.fps))
            if self.dataset != None:
                print('mae:{}'.format(self.mae_v))
                with open(dump_path, 'a') as f:
                    f.write('\nmae:{}'.format(self.mae_v))
        
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            mask = self.alphamaps[idx].permute(1,2,0).cpu().numpy()
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, viewpoint_cam.image_name + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, viewpoint_cam.image_name + ".png"))
            save_img_u8(self.diffuse[idx].permute(1,2,0).cpu().numpy(), os.path.join(diffuse_path, viewpoint_cam.image_name + ".png"))
            save_img_u8(self.specular[idx].permute(1,2,0).cpu().numpy(), os.path.join(specualr_path, viewpoint_cam.image_name + ".png"))
            save_img_u8(self.roughness[idx].repeat(3,1,1).permute(1,2,0).cpu().numpy(), os.path.join(roughness_path, viewpoint_cam.image_name + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth', viewpoint_cam.image_name + ".tiff"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal', viewpoint_cam.image_name + ".png"))
            # save_img_u8((self.ref_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5)*mask+(1-mask), os.path.join(vis_path, 'ref_normal', viewpoint_cam.image_name + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal', viewpoint_cam.image_name + ".png"))
            # torchvision.utils.save_image(self.normals[idx] * 0.5 + 0.5, os.path.join(vis_path, 'torch_normal', viewpoint_cam.image_name + ".png"))
            # torchvision.utils.save_image(self.depth_normals[idx] * 0.5 + 0.5, os.path.join(vis_path, 'torch_depth_normal', viewpoint_cam.image_name + ".png"))
            if self.ref_w:
                save_img_u8(self.ref_w[idx].repeat(3,1,1).permute(1,2,0).cpu().numpy(), os.path.join(vis_path, 'ref_w', viewpoint_cam.image_name + ".png"))
                save_img_u8(self.out_w[idx].repeat(3,1,1).permute(1,2,0).cpu().numpy(), os.path.join(vis_path, 'out_w', viewpoint_cam.image_name + ".png"))
    
    @torch.no_grad()
    def export_image_video(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            mask = self.alphamaps[idx].permute(1,2,0).cpu().numpy()
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_img_u8(((self.normals[idx]).permute(1,2,0).cpu().numpy() * 0.5 + 0.5)*mask+(1-mask), os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))

    
def get_database_eval_points(mesh_path):
    if os.path.exists(mesh_path):
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        return np.asarray(pcd.points)
    else:
        print("Mesh path doesn't exit.")
        
def project_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    mask0 = (np.abs(dpt)<1e-4) & (np.abs(dpt)>0)
    if np.sum(mask0)>0: dpt[mask0]=1e-4
    mask1=(np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1)>0: dpt[mask1]=-1e-4
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

        
def rasterize_depth_map(mesh,pose,K,shape):
    import nvdiffrast.torch as dr
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pts, depth = project_points(vertices,pose,K)
    # normalize to projection
    h, w = shape
    pts[:,0]=(pts[:,0]*2-w)/w
    pts[:,1]=(pts[:,1]*2-h)/h
    near, far = 5e-1, 1e2
    z = (depth-near)/(far-near)
    z = z*2 - 1
    pts_clip = np.concatenate([pts,z[:,None]],1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip,torch.ones_like(pts_clip[...,0:1])],1).unsqueeze(0)
    ctx = dr.RasterizeGLContext()
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w)) # [1,h,w,4]
    depth = (rast[0,:,:,2]+1)/2*(far-near)+near
    mask = rast[0,:,:,-1]!=0
    return depth.cpu().numpy(), mask.cpu().numpy().astype(bool)

def mask_depth_to_pts(mask,depth,K,rgb=None):
    hs,ws=np.nonzero(mask)
    depth=depth[hs,ws]
    pts=np.asarray([ws,hs,depth],np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    if rgb is not None:
        return np.dot(pts, np.linalg.inv(K).transpose()), rgb[hs,ws]
    else:
        return np.dot(pts, np.linalg.inv(K).transpose())
    
def pose_apply(pose,pts):
    return transform_points_pose(pts, pose)

def transform_points_pose(pts, pose):
    R, t = pose[:, :3], pose[:, 3]
    if len(pts.shape)==1:
        return (R @ pts[:,None] + t[:,None])[:,0]
    return pts @ R.T + t[None,:]

def pose_inverse(pose):
    R = pose[:,:3].T
    t = - R @ pose[:,3:]
    return np.concatenate([R,t],-1)
        
def get_mesh_eval_points(scene):
    eval_mesh_path = os.path.join(scene.model_path, "train/ours_"+str(scene.loaded_iter), "fuse_post.ply")
    mesh = trimesh.load_mesh(eval_mesh_path)
    pbar = tqdm(len(scene.getTrainCameras()))
    pts_pr = []
    for i, viewpoint_cam in tqdm(enumerate(scene.getTrainCameras())):
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width
        fx, fy, cx, cy = viewpoint_cam.Fx, viewpoint_cam.Fy, viewpoint_cam.Cx, viewpoint_cam.Cy
        K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ])
        pose = viewpoint_cam.world_view_transform.transpose(0, 1).cpu().numpy()  # c2w
        pose = pose[:3, :]
        depth_pr, mask_pr = rasterize_depth_map(mesh, pose, K, (h, w))
        pts_ = mask_depth_to_pts(mask_pr, depth_pr, K)
        pose = pose_inverse(pose)
        pts_pr.append(pose_apply(pose, pts_))
        pbar.update(1)

    pts_pr = np.concatenate(pts_pr, 0).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_pr)
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    return np.asarray(downpcd.points,np.float32)

def nearest_dist(pts0, pts1, batch_size=512):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc='evaluting...'):
        dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
        dists.append(torch.min(dist,1)[0])
    dists = torch.cat(dists,0)
    return dists.cpu().numpy()