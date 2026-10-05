import torch
from scene import Gaussians
from tqdm import tqdm 
from render import Renderer_view
from utils.loss_utils import calc_mse, calc_psnr
from pytorch_msssim import ms_ssim

import evo
import copy
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
import matplotlib.pyplot as plt
import evo.tools.plot
import numpy as np
import os 
import json
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
loss_fn_alex = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()

import cv2
import open3d as o3d 
from utils.eval_mesh import evaluate_reconstruction
from pathlib import Path
from datasets import load_dataset_config, get_dataset


def save_mesh_checkpoint(config: dict, gaussians: Gaussians, w2cs, gt_w2cs, eval_dir:str="tmp"):
    
    cp_idx = len(w2cs)
    device = torch.device(config["primary_device"])
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=False,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
        
    num_frames = len(dataset)   
    
    os.makedirs(os.path.join(eval_dir, "check_point_mesh"), exist_ok=True)
    intrinsics_list = config['cameras']['intrinsics']
    width = config['cameras']['width']
    height = config['cameras']['height']
    
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        width, height, intrinsics_list[0][0], intrinsics_list[1][1], intrinsics_list[0][2], intrinsics_list[1][2]         
    )
    
    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    _, _, _, first_c2w = dataset[0]
    first_w2c = torch.linalg.inv(first_c2w).detach().cpu().numpy()
        
    for time_idx in tqdm(range(cp_idx), total=cp_idx, desc='check_point_mesh: '):
        gt_color, gt_depth, intrinsics, gt_pose = dataset[time_idx] 
        gt_color = (gt_color / 255).permute(2, 0, 1)
        gt_depth = (gt_depth).permute(2, 0, 1)
        pred_w2c = w2cs[time_idx]
        
        with torch.no_grad():
            render_pkg = Renderer_view(config, gaussians, pred_w2c)
            render_color = torch.clamp(render_pkg['render_color'], 0.0, 1.0)
            render_depth = torch.nan_to_num(render_pkg['render_depth'] / render_pkg['render_alpha'], 0.0, 0.0)
            
      
            if time_idx % config['eval']['mesh_interval'] == 0:
                color_np = torch.clamp(render_color.permute(1, 2, 0), 0., 1.).contiguous().detach().cpu().numpy() * 255
                color_np = color_np.astype(np.uint8)
                depth_np = render_depth.squeeze(0).detach().cpu().numpy()
                est_w2c = pred_w2c.detach().cpu().numpy()
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(color_np)),
                    o3d.geometry.Image(depth_np),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                if config['data']['dataset_name'] == "scannetpp":
                    P = torch.tensor(
                        [
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ]).float().numpy()
                else:
                    P = np.eye(4)
                volume.integrate(rgbd, intrinsics_o3d,  est_w2c @ first_w2c @ P)
        
    o3d_mesh = volume.extract_triangle_mesh()
    compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                            scale / 512.0, -2.5 * scale / 512.0)
    o3d_mesh = o3d_mesh.translate(compensate_vector)
    file_name = os.path.join(eval_dir, "check_point_mesh", f"checkpoint_mesh_{time_idx:05d}.ply")
    o3d.io.write_triangle_mesh(file_name, o3d_mesh)

    
# for scannetpp 
def eval_nvs(config: dict, gaussians: Gaussians, eval_dir):
    device = torch.device(config["primary_device"])
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
  
    dataset_config["use_train_split"] = False
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    l1_list = []
    rmse_list = []
    
    os.makedirs(eval_dir, exist_ok=True)
    
    save_renders = True
    render_rgb_dir = os.path.join(eval_dir, "rendering/rgb")
    render_depth_dir = os.path.join(eval_dir, "rendering/depth")
    render_differ_dir = os.path.join(eval_dir, "rendering/diff")
    if save_renders:
        os.makedirs(render_rgb_dir, exist_ok=True)
        os.makedirs(render_depth_dir, exist_ok=True)
        os.makedirs(render_differ_dir, exist_ok=True)
    num_frames = len(dataset)
    for time_idx in tqdm(range(num_frames), total=num_frames, desc='Eval: '):
        gt_color, gt_depth, intrinsics, gt_pose = dataset[time_idx] 
        gt_color = (gt_color / 255).permute(2, 0, 1)
        gt_depth = (gt_depth).permute(2, 0, 1)
        pred_w2c = torch.linalg.inv(gt_pose)
        
        with torch.no_grad():
            render_pkg = Renderer_view(config, gaussians, pred_w2c)
            render_color = torch.clamp(render_pkg['render_color'], 0.0, 1.0)
            render_depth = torch.nan_to_num(render_pkg['render_depth'] / render_pkg['render_alpha'], 0.0, 0.0)
            
            if save_renders:
                
                viz_render_im = render_color
                viz_render_im = viz_render_im.detach().cpu().permute(1, 2, 0).numpy()
                rastered_depth_viz = render_depth.detach()
                vmin = 0
                vmax = 6
                viz_render_depth = rastered_depth_viz[0].detach().cpu().numpy()
                normalized_depth = np.clip((viz_render_depth - vmin) / (vmax - vmin), 0, 1)
                depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(render_rgb_dir, "GauS_{:04d}.png".format(time_idx)), cv2.cvtColor(viz_render_im*255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(render_depth_dir, "GauS_{:04d}.png".format(time_idx)), depth_colormap)
                
                dmin = 0
                dmax = 0.02
                diff = torch.abs(render_depth - gt_depth)
                diff[gt_depth < 1e-4] = 0
                diff = diff.squeeze(0).detach().cpu().numpy()
                normalized_diff = np.clip((diff - dmin) / (dmax - dmin), 0, 1)
                diff_colormap = cv2.applyColorMap((normalized_diff * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                cv2.imwrite(os.path.join(render_differ_dir, "GauS_{:04d}.png".format(time_idx)), diff_colormap)
               
            valid_depth_mask = (gt_depth > 0)
            weighted_im = render_color * valid_depth_mask
            weighted_gt_im = gt_color * valid_depth_mask
            rastered_depth = render_depth * valid_depth_mask

            psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                           data_range=1.0, size_average=True)
            lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                       torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

            psnr_list.append(psnr.cpu().numpy())
            ssim_list.append(ssim.cpu().numpy())
            lpips_list.append(lpips_score)
            
            diff_depth_rmse = ((rastered_depth - gt_depth) ** 2)
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = torch.sqrt(diff_depth_rmse.sum() / valid_depth_mask.sum())
            diff_depth_l1 = torch.abs((rastered_depth - gt_depth))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
            rmse_list.append(rmse.cpu().numpy())
            l1_list.append(depth_l1.cpu().numpy())
            

    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean() 
    
    result = {
        'PSNR: ': float(avg_psnr),
        'SSIM: ': float(avg_ssim),
        'LPIPS: ': float(avg_lpips),
        'Depth RMSE: ': float(avg_rmse),
        'Depth L1: ': float(avg_l1),
    }
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
        
    with open(os.path.join(eval_dir, "nvs_result.json"), 'w') as f:
        json.dump(result, f, indent = 2)
        
    
def eval_final(config: dict, gaussians: Gaussians, w2cs, gt_w2cs, eval_dir):
    
    device = torch.device(config["primary_device"])
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
            gradslam_data_cfg = {}
            gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=False,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
        
    num_frames = len(dataset)   
    notbad = [id for id in range(len(gt_w2cs)) if not (torch.isnan(gt_w2cs[id]).any() or torch.isinf(gt_w2cs[id]).any())]
    nobad_w2cs = [w2cs[id] for id in notbad] # w2cs[notbad]
    nobad_gt_w2cs = [gt_w2cs[id] for id in notbad] #  gt_w2cs[notbad]
    
    pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in nobad_w2cs]
    gt_pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in nobad_gt_w2cs]
    traj_ref = PosePath3D(poses_se3=gt_pose_w2cs)
    traj_est = PosePath3D(poses_se3=pose_w2cs)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False)
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    l1_list = []
    rmse_list = []
    
    os.makedirs(eval_dir, exist_ok=True)
    
    save_renders = config['eval']['save_renders']
    eval_mesh = config['eval']['eval_mesh']

    render_rgb_dir = os.path.join(eval_dir, "rendering/rgb")
    render_depth_dir = os.path.join(eval_dir, "rendering/depth")
    render_differ_dir = os.path.join(eval_dir, "rendering/diff")
    if save_renders:
        os.makedirs(render_rgb_dir, exist_ok=True)
        os.makedirs(render_depth_dir, exist_ok=True)
        os.makedirs(render_differ_dir, exist_ok=True)
        
    if eval_mesh:
        
        # load first_pose 
        # pose_path = Path(config['data']['basedir']) / config['data']['sequence'] / "traj.txt"
        # with open(pose_path, 'r') as f:
        #     line = f.readline()
        # first_w2c = np.linalg.inv(np.array(list(map(float, line.split()))).reshape(4, 4))
        
        os.makedirs(os.path.join(eval_dir, "mesh"), exist_ok=True)
        intrinsics_list = config['cameras']['intrinsics']
        width = config['cameras']['width']
        height = config['cameras']['height']
        
        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsics_list[0][0], intrinsics_list[1][1], intrinsics_list[0][2], intrinsics_list[1][2]         
        )
        
        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * scale / 512.0,
            sdf_trunc=0.04 * scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    if eval_mesh:
        _, _, _, first_c2w = dataset[0]
        first_w2c = torch.linalg.inv(first_c2w).detach().cpu().numpy()
        
    for time_idx in tqdm(range(num_frames), total=num_frames, desc='Eval: '):
        gt_color, gt_depth, intrinsics, gt_pose = dataset[time_idx] 
        gt_color = (gt_color / 255).permute(2, 0, 1)
        gt_depth = (gt_depth).permute(2, 0, 1)
        pred_w2c = w2cs[time_idx]
        
        with torch.no_grad():
            render_pkg = Renderer_view(config, gaussians, pred_w2c)
            render_color = render_pkg['render_color']
            render_depth = render_pkg['render_depth'] 
            
            if save_renders:
                color_np = torch.clamp(render_color.permute(1, 2, 0), 0., 1.).contiguous().detach().cpu().numpy() * 255
                color_np = color_np.astype(np.uint8)
                depth_np = render_depth.squeeze(0).detach().cpu().numpy()
                # cv2.imwrite(os.path.join(eval_dir, "renders", f"color_{time_idx}.png"), color_np)
                # cv2.imwrite(os.path.join(eval_dir, "renders", f"depth_{time_idx}.tiff"), depth_np)  
                vmin = 0
                vmax = 6
                normalized_depth = np.clip((depth_np - vmin) / (vmax - vmin), 0, 1)
                depth_colormap = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(render_rgb_dir, "GauS_{:04d}.png".format(time_idx)), cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(render_depth_dir, "GauS_{:04d}.png".format(time_idx)), depth_colormap)
                
                dmin = 0
                dmax = 0.02
                diff = torch.abs(render_depth - gt_depth)
                diff = diff.squeeze(0).detach().cpu().numpy()
                normalized_diff = np.clip((diff - dmin) / (dmax - dmin), 0, 1)
                diff_colormap = cv2.applyColorMap((normalized_diff * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
                cv2.imwrite(os.path.join(render_differ_dir, "differ_{:04d}.png".format(time_idx)), diff_colormap)

            if eval_mesh and time_idx % config['eval']['mesh_interval'] == 0:
                color_np = torch.clamp(render_color.permute(1, 2, 0), 0., 1.).contiguous().detach().cpu().numpy() * 255
                color_np = color_np.astype(np.uint8)
                depth_np = render_depth.squeeze(0).detach().cpu().numpy()
                est_w2c = pred_w2c.detach().cpu().numpy()
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(color_np)),
                    o3d.geometry.Image(depth_np),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                if config['data']['dataset_name'] == "scannetpp":
                    P = torch.tensor(
                        [
                            [0, 1, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ]).float().numpy()
                else:
                    P = np.eye(4)
                volume.integrate(rgbd, intrinsics_o3d,  est_w2c @ first_w2c @ P)
                
            valid_depth_mask = (gt_depth > 0)
            weighted_im = render_color * valid_depth_mask
            weighted_gt_im = gt_color * valid_depth_mask
            rastered_depth = render_depth * valid_depth_mask

            psnr = calc_psnr(weighted_im, weighted_gt_im).mean()
            ssim = ms_ssim(weighted_im.unsqueeze(0).cpu(), weighted_gt_im.unsqueeze(0).cpu(), 
                           data_range=1.0, size_average=True)
            lpips_score = loss_fn_alex(torch.clamp(weighted_im.unsqueeze(0), 0.0, 1.0),
                                       torch.clamp(weighted_gt_im.unsqueeze(0), 0.0, 1.0)).item()

            psnr_list.append(psnr.cpu().numpy())
            ssim_list.append(ssim.cpu().numpy())
            lpips_list.append(lpips_score)
            
            diff_depth_rmse = ((rastered_depth - gt_depth) ** 2)
            diff_depth_rmse = diff_depth_rmse * valid_depth_mask
            rmse = torch.sqrt(diff_depth_rmse.sum() / valid_depth_mask.sum())
            diff_depth_l1 = torch.abs((rastered_depth - gt_depth))
            diff_depth_l1 = diff_depth_l1 * valid_depth_mask
            depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()
            rmse_list.append(rmse.cpu().numpy())
            l1_list.append(depth_l1.cpu().numpy())
            

    psnr_list = np.array(psnr_list)
    rmse_list = np.array(rmse_list)
    l1_list = np.array(l1_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)
    avg_psnr = psnr_list.mean()
    avg_rmse = rmse_list.mean()
    avg_l1 = l1_list.mean()
    avg_ssim = ssim_list.mean()
    avg_lpips = lpips_list.mean() 
    
    result = {
        'PSNR: ': float(avg_psnr),
        'SSIM: ': float(avg_ssim),
        'LPIPS: ': float(avg_lpips),
        'Depth RMSE: ': float(avg_rmse),
        'Depth L1: ': float(avg_l1),
        'ATE RMSE: ': float(ape_stat), 
    }
    print("Final Result ATE RMSE: {:.2f} cm".format(ape_stat * 100))   
    print("Average PSNR: {:.2f}".format(avg_psnr))
    print("Average Depth RMSE: {:.2f} cm".format(avg_rmse*100))
    print("Average Depth L1: {:.2f} cm".format(avg_l1*100))
    print("Average MS-SSIM: {:.3f}".format(avg_ssim))
    print("Average LPIPS: {:.3f}".format(avg_lpips))

    np.savetxt(os.path.join(eval_dir, "psnr.txt"), psnr_list)
    np.savetxt(os.path.join(eval_dir, "rmse.txt"), rmse_list)
    np.savetxt(os.path.join(eval_dir, "l1.txt"), l1_list)
    np.savetxt(os.path.join(eval_dir, "ssim.txt"), ssim_list)
    np.savetxt(os.path.join(eval_dir, "lpips.txt"), lpips_list)
    
    if eval_mesh:
        o3d_mesh = volume.extract_triangle_mesh()
        
        compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                             scale / 512.0, -2.5 * scale / 512.0)
        o3d_mesh = o3d_mesh.translate(compensate_vector)
        
        file_name = os.path.join(eval_dir, "mesh", "final_mesh.ply")
        o3d.io.write_triangle_mesh(file_name, o3d_mesh)
        scene_name = config['data']['sequence']
        
        if config['data']['dataset_name'] == 'Replica':
            meshdir = config['data']['meshdir']
            evaluate_reconstruction(Path(file_name),
                                    f"{meshdir}/{scene_name}.ply",
                                    f"{meshdir}/{scene_name}_pc_unseen.npy",
                                    Path(eval_dir))
        
        if config['data']['dataset_name'] == 'scannetpp':
            evaluate_reconstruction(Path(file_name),
                                    f"./data/scannetpp/{scene_name}/scans/mesh_aligned_0.05.ply",
                                    f"", # not support depth_l1
                                    Path(eval_dir))
        
    with open(os.path.join(eval_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent = 2)

    return result