import copy
import math
import numpy as np
from tqdm import tqdm
import os
import json

import torch
import torchvision
import torch.nn.functional as F

from diff_surfel_2dgs import GaussianRasterizationSettings as GaussianRasterizationSettings_2dgs
from diff_surfel_2dgs import GaussianRasterizer as GaussianRasterizer_2dgs
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from diff_surfel_rasterization_feature import GaussianRasterizationSettings as GaussianRasterizationSettings_feature
from diff_surfel_rasterization_feature import GaussianRasterizer as GaussianRasterizer_feature


from diff_surfel_rasterization_real import GaussianRasterizationSettings as GaussianRasterizationSettings_real
from diff_surfel_rasterization_real import GaussianRasterizer as GaussianRasterizer_real

from scene.gaussian_model import GaussianModel
from scene.ref_gaussian_model import RefGaussianModel

from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.graphics_utils import fov2focal

from utils.color_utils import *
from utils.sph_utils import *

import nvdiffrast.torch as dr

def get_outside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

def get_inside_msk(xyz, ENV_CENTER, ENV_RADIUS):
    return torch.sum((xyz - ENV_CENTER[None])**2, dim=-1) <= ENV_RADIUS**2

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings_black = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer_black = GaussianRasterizer(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    rets =  {}

    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness  # (N, 1)
    
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = gs_albedo,
        language_feature_precomp = gs_roughness,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    with torch.no_grad():
        select_index = (render_alpha.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
    
    roughness_map = out_ts.clone()
    diff_light = albedo_map
    
    output_diff = linear2srgb(diff_light)
    
    rets.update({
        'output_diff': output_diff,
        "diff_light": diff_light,
        
        "select_index": select_index,

        "rend_roughness": roughness_map,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 
    

    return rets


def render_ref(viewpoint_camera, pc : RefGaussianModel, pipe, bg_color : torch.Tensor, render_pkg, scaling_modifier = 1.0, XYZ=[0,1,2]):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings_black = GaussianRasterizationSettings_feature(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer_black = GaussianRasterizer_feature(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    rets =  {}

    gs_feature = pc.get_language_feature  # (N, 4)
    
    input_ts = torch.cat([gs_feature], dim=-1)
    
    albedo_map, out_ts, radii, allmap = rasterizer_black(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = torch.zeros([gs_feature.shape[0], 3], device="cuda"),
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    
    viewdirs = viewpoint_camera.rays_d
    normals = render_pkg["rend_normal"].permute(1,2,0)
    wo = F.normalize(reflect(-viewdirs, normals), dim=-1)
    
    indi_feature_map = out_ts.permute(1,2,0)  # (H, W, 4)
    
    roughness_map = render_pkg["rend_roughness"].permute(1,2,0)

    select_index = render_pkg["select_index"]
    
    wo = wo.reshape(-1, 3)[select_index]
    normals = normals.reshape(-1, 3)[select_index]
    roughness_map = roughness_map.reshape(-1, 1)[select_index]
    
    indi_feature_dirc = indi_feature_map.reshape(-1, pc.gsfeat_dim)[select_index]
    
    ''' Sph-Mip '''
    wo_xy = (cart2sph(wo.reshape(-1, 3)[..., XYZ])[..., 1:] / torch.Tensor([[np.pi, 2*np.pi]]).cuda())[..., [1,0]]
    wo_xyz = torch.stack([wo_xy[:, None, :]], dim=0,)  # (1, N_pix, 1, 2)
    
    spec_level = roughness_map.reshape(-1, 1)  # (N_pix, 1)

    spec_feat = pc.dir_encoding(wo_xyz, spec_level.view(-1, 1), index=0).reshape(-1, pc.sph_dim)  # (N_pix, 16)
    spec_feat_dirc = spec_feat.reshape(-1, pc.sph_dim)  # (N_pix, 16)
    
    
    #####################################################################################################################

    cosvn = (-viewdirs.reshape(-1, 3)[select_index] * normals).sum(dim=-1, keepdim=True).clamp(0, 1)
    
    # specular color
    input_mlp = torch.cat([spec_feat_dirc, indi_feature_dirc, roughness_map, cosvn], -1)
    
    mlp_output = pc.light_mlp(input_mlp).float()  # (N_pix, 3)
    spec_light = torch.exp(torch.clamp(mlp_output, max=5.0))
    
    #####################################################################################################################
    
    out_spec_light = torch.zeros(image_height, image_width, 3).cuda()
    out_spec_light.reshape(-1, 3)[select_index] = spec_light
    out_spec_light = out_spec_light.permute(2,0,1)
    
    output_spec = torch.zeros(image_height, image_width, 3).cuda()
    output_spec.reshape(-1, 3)[select_index] = linear2srgb(spec_light)
    output_spec = output_spec.permute(2,0,1)
    
    rets.update({
        'output_spec': output_spec,
        "spec_light": out_spec_light,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 
            

    return rets


def render_real(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, ENV_CENTER=None, ENV_RADIUS=None, XYZ=[0,1,2]):
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    image_height = int(viewpoint_camera.image_height)
    image_width = int(viewpoint_camera.image_width)

    raster_settings_black = GaussianRasterizationSettings_real(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color*0.0,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    
    rasterizer_black = GaussianRasterizer_real(raster_settings=raster_settings_black)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features
    
    gs_albedo = pc.get_albedo
    gs_roughness = pc.get_roughness
    
    gs_in = torch.ones_like(gs_roughness)
    gs_in[get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS)] = 0.0
    
    gs_out = torch.zeros_like(gs_roughness)
    gs_out[get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS)] = 1.0

    input_ts = torch.cat([gs_roughness, gs_albedo, gs_in, gs_out], dim=-1)
    
    rendered_image, out_ts, radii, allmap = rasterizer_black(  # albedo_map=[3, H, W], out_ts=[H, W, N]
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        language_feature_precomp = input_ts,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    
    rets =  {
        "render": rendered_image,
    }

    render_alpha = allmap[1:2]

    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    render_normal = F.normalize(render_normal, dim=0)

    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / (render_alpha))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]

    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    surf_normal = surf_normal * (render_alpha).detach()

    #####################################################################################################################
    
    out_ts = out_ts.permute(1,2,0)
    
    roughness_map = out_ts[..., :1]
    albedo_map = out_ts[..., 1:4]
    
    in_map = out_ts[..., 4:5]
    
    ref_w = (out_ts[..., 4:5].permute(2,0,1)).detach()
    out_w = 1-ref_w
    
    out_roughness = roughness_map.permute(2,0,1)
    
    with torch.no_grad():
        select_index = (in_map.reshape(-1,) > 0.05).nonzero(as_tuple=True)[0]
        
    diff_light = albedo_map.permute(2,0,1) + (1-render_alpha) * bg_color.view(3, 1, 1)
    
    output_diff = linear2srgb(diff_light)
    
    rets.update({  # [N, H, W]
        "diff_light": diff_light, 
        'output_diff': output_diff,
        'ref_w': ref_w,
        'out_w': out_w,
        'ref_index': get_outside_msk(pc.get_xyz, ENV_CENTER, ENV_RADIUS),
        
        "rend_roughness": out_roughness,
        
        "select_index": select_index,

        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }) 

    return rets