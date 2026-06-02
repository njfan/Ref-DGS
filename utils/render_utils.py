# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm

import torch

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]
  t_mean = t.mean(axis=0)
  t = t - t_mean

  eigval, eigvec = np.linalg.eig(t.T @ t)
  # Sort eigenvectors in order of largest to smallest eigenvalue.
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

  # Flip coordinate system if z component of y-axis is negative
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform
  # points = np.random.rand(3,100)
  # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
  # (poses_recentered @ points_h)[0]
  # (transform @ pad_poses(poses) @ points_h)[0,:3]
  # import pdb; pdb.set_trace()

  # # Just make sure it's it in the [-1, 1]^3 cube
  # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  # poses_recentered[:, :3, 3] *= scale_factor
  # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  # return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset
  high = sc + offset
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
        z_variation * (z_low[2] + (z_high - z_low)[2] *
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
  positions = get_positions(theta)

  #if const_speed:

  # # Resample theta angles so that the velocity is closer to constant.
  # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
  # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
  # positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)
  avg_up = avg_up / np.linalg.norm(avg_up)
  ind_up = np.argmax(np.abs(avg_up))
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

  return np.stack([viewmatrix(p - center, up, p) for p in positions])

import torch.nn.functional as F
from utils.camera_utils import get_rays
def generate_path(viewpoint_cameras, n_frames=480):
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  # generate new poses
  new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
  # warp back to orignal scale
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []
  for c2w in new_poses:
      c2w = c2w @ np.diag([1, -1, -1, 1])
      w2c = np.linalg.inv(c2w)
      R = w2c[:3, :3]
      T = w2c[:3, 3]
      cam = copy.deepcopy(viewpoint_cameras[0])
      cam.image_height = int(cam.image_height / 2) * 2
      cam.image_width = int(cam.image_width / 2) * 2
      cam.R = torch.from_numpy(R).float().cuda()
      cam.T = torch.from_numpy(T).float().cuda()
      cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
      cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
      cam.camera_center = cam.world_view_transform.inverse()[3, :3]
      cam.rays_o, rays_d = get_rays(cam.image_width, cam.image_height, cam.FoVx, cam.FoVx, cam.R.cpu(), cam.T.cpu())
      cam.rays_d = F.normalize(rays_d.cuda(), dim=-1)
      traj.append(cam)

  return traj

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
  """Creates videos out of the images saved to disk."""
  # Last two parts of checkpoint path are experiment name and scene name.
  video_prefix = f'{out_name}'

  zpad = max(5, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  os.makedirs(base_dir, exist_ok=True)
  render_dist_curve_fn = np.log
  
  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')
  depth_frame = load_img(depth_file)
  shape = depth_frame.shape
  p = 3
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')

  video_kwargs = {
      'shape': shape[:2],
      'codec': 'h264',
      'fps': 60,
      'crf': 18,
  }
  
  for k in ['depth', 'normal', 'color']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
    input_format = 'gray' if k == 'alpha' else 'rgb'
    

    file_ext = 'png' if k in ['color', 'normal'] else 'tiff'
    idx = 0

    if k == 'color':
      file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
    else:
      file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

    if not os.path.exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in tqdm(range(num_frames)):
        # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        if k == 'color':
          img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
        else:
          img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

        if not os.path.exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = load_img(img_file)
        if k in ['color', 'normal']:
          img = img / 255.
        elif k.startswith('depth'):
          img = render_dist_curve_fn(img)
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          img = cm.get_cmap('turbo')(img)[..., :3]

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)
        idx += 1

os.environ['PYOPENGL_PLATFORM'] = 'egl' 
import pyrender        
import trimesh

def create_mesh_videos(mesh_path, cam_traj, save_path):
    print(f"Loading mesh from {mesh_path}...")
    
    # 1. 加载 Mesh
    mesh = trimesh.load(mesh_path)
    
    # 清理颜色，确保不被原始颜色干扰
    # if hasattr(mesh.visual, 'vertex_colors'): mesh.visual.vertex_colors = None
    # if hasattr(mesh.visual, 'face_colors'): mesh.visual.face_colors = None
    
    # try:
    #     if mesh.visual.kind == 'face': mesh.fix_normals()
    # except: pass

    # 2. 设置材质 (双面渲染，防止法线反向导致全黑)
    # Ref-NeRF 的物体通常比较闪亮 (Shiny)，我们调高 metallic
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.6, 0.6, 0.6, 1.0], 
        metallicFactor=0.0,  # 增加金属感
        roughnessFactor=0.3, # 表面光滑
        doubleSided=True     # 【核心】防止背面剔除
    )
    
    mesh_node = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    # 场景：增加环境光，防止死黑
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.1, 0.1, 0.1])
    scene.add(mesh_node)

    # 准备渲染
    H, W = cam_traj[0].image_height, cam_traj[0].image_width
    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    
    frames = []
    print(f"Rendering {len(cam_traj)} frames...")
    
    for idx, cam in enumerate(tqdm(cam_traj)):
        # 3. 内参
        camera = pyrender.IntrinsicsCamera(
            fx=cam.Fx, fy=cam.Fy, cx=cam.Cx, cy=cam.Cy, 
            znear=cam.znear, zfar=cam.zfar
        )
        # 4. 外参 (Pose)
        # 获取 W2C
        w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
        # 获取 C2W
        c2w = np.linalg.inv(w2c)
        
        # 【关键修正】应用 OpenCV -> OpenGL 翻转
        c2w = c2w @ np.diag([1, -1, -1, 1])
        
        # 5. 确保相机节点添加
        cam_node = scene.add(camera, pose=c2w)
        
        # 6. 灯光系统
        # 使用点光源 (PointLight) 绑定在相机上，类似矿工头灯
        # 强度设为 10.0 (根据你的距离 4.0 左右，这个强度适中)
        #light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light_node = scene.add(light, pose=c2w)

        # 渲染
        color, _ = r.render(scene)
        frames.append(color)

        # 清理
        scene.remove_node(cam_node)
        scene.remove_node(light_node)

    r.delete()

    # 7. 保存视频
    print("Saving mesh video...")
    video_path = os.path.join(save_path, f"render_mesh_shiny.mp4")
    
    with media.VideoWriter(
        video_path, shape=(H, W), codec='h264', fps=60, crf=18
    ) as writer:
        for frame in frames:
            writer.add_image(frame)
    
    print(f"Mesh video saved to {video_path}")

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')