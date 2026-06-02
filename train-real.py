import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import uuid
import torch
import numpy as np
from tqdm import tqdm
import random

from random import randint
from scene import Scene, GaussianModel
from utils.loss_utils import l1_loss, ssim, entropy_loss, binary_cross_entropy, tv_loss, depth_prior_loss
from utils.general_utils import safe_state, save_image_real
from gaussian_renderer import *

from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from fused_ssim import fused_ssim as fast_ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train-real.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)
    gaussians = GaussianModel(dataset.sh_degree, dataset)
    ref_gaussians = RefGaussianModel(dataset.sh_degree, dataset)
    
    scene = Scene(dataset, gaussians, ref_gaussians, resolution_scales=[1.0])
    
    gaussians.training_setup(opt)
    ref_gaussians.training_setup(opt)
    
    
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    ###########################################################################
    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    print('Training set length', len(viewpoint_stack))
        
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    dataset_name = dataset.source_path.split('/')[-1]

    ENV_CENTER = torch.tensor([float(c) for c in dataset.env_scope_center], device='cuda')
    ENV_RADIUS = dataset.env_scope_radius
    XYZ = [int(float(c)) for c in dataset.xyz_axis]
    
    print(ENV_CENTER, ENV_RADIUS, XYZ)
        
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        ref_gaussians.update_learning_rate(iteration)
        

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        data_idx = np.random.randint(len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack[data_idx]
        
        bg = torch.zeros((3), device="cuda") # 真实数据集
        
        ITER = dataset.init_until_iter
        
        render_pkg = render_real(viewpoint_cam, gaussians, pipe, bg, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        ref_w = render_pkg["ref_w"]
        out_w = render_pkg["out_w"]
        
        gt_image = viewpoint_cam.original_image.cuda()
            
        loss = 0.0
        
        # regularization
        lambda_normal = 0.05 if iteration > ITER else 0.0
        lambda_dist = 0.0 if iteration > ITER else 0.0
        
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        loss = loss + dist_loss + normal_loss
        
        
        if iteration > ITER:
            ref_render_pkg = render_ref(viewpoint_cam, ref_gaussians, pipe, bg, render_pkg, XYZ=XYZ)
            
            ref_viewspace_point_tensor = ref_render_pkg["viewspace_points"]
            ref_visibility_filter = ref_render_pkg["visibility_filter"]
            ref_radii = ref_render_pkg["radii"]
        
            pbr_rgb = get_final_color(render_pkg, ref_render_pkg, bg)
            
            Ll1 = l1_loss(pbr_rgb, gt_image)
            loss_pbr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fast_ssim(pbr_rgb.unsqueeze(0), gt_image.unsqueeze(0)))
            loss += loss_pbr
            
            Ll1 = l1_loss(image*out_w, gt_image*out_w)
            loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fast_ssim((image*out_w).unsqueeze(0), (gt_image*out_w).unsqueeze(0)))
            loss += loss_rgb
            
            gt_mask = viewpoint_cam.gt_alpha_mask * ref_w # [1, H, W]
            
            density_loss = entropy_loss(gaussians.get_opacity[visibility_filter])
            loss += density_loss * 0.001
            
            # prior
            if iteration < opt.vggt_until_iter:
                cosine_term = 1.0 - ((rend_normal) * (viewpoint_cam.vggt_normal)).sum(0).clamp(-1.0, 1.0)
                l1_term = (rend_normal - viewpoint_cam.vggt_normal).abs().sum(0)

                vggt_normal_loss = opt.vggt_weight * ((cosine_term + 0.5 * l1_term)).sum() / torch.sum(gt_mask)

                vggt_depth_loss = opt.vggt_weight * depth_prior_loss(render_pkg["surf_depth"]*gt_mask, viewpoint_cam.vggt_depth*gt_mask, 
                                                                     viewpoint_cam.vggt_depth_conf, gt_mask>0.5)
                
                loss += vggt_normal_loss + vggt_depth_loss
            
            
        else:
            pbr_rgb = image

            Ll1 = l1_loss(pbr_rgb, gt_image)
            loss_pbr = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fast_ssim(pbr_rgb.unsqueeze(0), gt_image.unsqueeze(0)))
            loss += loss_pbr
            
            alpha_loss=torch.zeros_like(Ll1)
            vggt_normal_loss=torch.zeros_like(Ll1)
            vggt_depth_loss=torch.zeros_like(Ll1)
        
        # loss
        total_loss = loss
        total_loss.backward()
        iter_end.record()
        
        if iteration==ITER+1 or (iteration>ITER and iteration%500 == 0):
            with torch.no_grad():     
                save_image_real(iteration, dataset.model_path, viewpoint_cam, render_pkg, ref_render_pkg, pbr_rgb)
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "GS-Points": f"{len(gaussians.get_xyz)}",
                    "REFGS-Points": f"{len(ref_gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict, refresh=False)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, Ll1, loss, loss_pbr, alpha_loss, dist_loss, normal_loss, vggt_normal_loss, vggt_depth_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene, render_real, render_ref, pipe, background, ENV_CENTER, ENV_RADIUS, XYZ)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.gs_densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    print(gaussians._opacity.data[torch.isnan(gaussians._opacity.data.mean(dim=-1))].shape)
                    
            if ITER < iteration < opt.refgs_densify_until_iter:
                ref_gaussians.max_radii2D[ref_visibility_filter] = torch.max(ref_gaussians.max_radii2D[ref_visibility_filter], ref_radii[ref_visibility_filter])
                ref_gaussians.add_densification_stats(ref_viewspace_point_tensor, ref_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    ref_gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    ref_gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                ref_gaussians.optimizer.step()
                ref_gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save((ref_gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "ref_gaussians.pth")

                
def prepare_output_and_logger(args):
    dataset_name = args.source_path.split('/')[-1]
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./output/ref-real/", dataset_name)
        
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, loss_pbr, alpha_loss, dist_loss, normal_loss, vggt_normal_loss, vggt_depth_loss,
                    elapsed, testing_iterations, scene : Scene, renderFunc, renderrefFunc, pipe, bg, ENV_CENTER, ENV_RADIUS, XYZ):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/loss_pbr', loss_pbr.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/alpha_loss', alpha_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/dist_loss', dist_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/vggt_normal_loss', vggt_normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/vggt_depth_loss', vggt_depth_loss.item(), iteration)
        
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg, ENV_CENTER=ENV_CENTER, ENV_RADIUS=ENV_RADIUS, XYZ=XYZ)
                    ref_render_pkg = renderrefFunc(viewpoint, scene.ref_gaussians, pipe, bg, render_pkg, XYZ=XYZ)
                    image = get_final_color(render_pkg, ref_render_pkg, bg)
                    #image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=
                        [15_000, 20_000, 25_000, 30_000]
                       )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=
                        [15_000, 20_000, 25_000, 30_000]
                       )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")