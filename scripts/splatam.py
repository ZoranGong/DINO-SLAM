import sys
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

from importlib.machinery import SourceFileLoader
import argparse
import torch
import random
from utils.common_utils import seed_everything, get_pointcloud
from datasets import get_dataset, load_dataset_config
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scene import Frame, Gaussians
from render import Renderer_tracking, Renderer_mapping
import time 

from utils.descriptor import GlobalDesc
import evo
import copy
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
import matplotlib.pyplot as plt
import evo.tools.plot
from slam.Loss import get_loss
from slam.Densify import add_new_gaussians, prune_gaussians
from scene import save_scence
from utils.eval import eval_final
import json 

class LocalMaps:
    
    def __init__(self, w2cs, gt_w2cs):
        self.w2cs = w2cs
        self.gt_w2cs = gt_w2cs 
    
    def get_w2cs(self, ):
        return self.w2cs 
    
    def get_gt_w2cs(self, ):
        return self.gt_w2cs

def rgbd_slam(config: dict):
    
    dataset_config = config["data"]
    # Get Device
    device = torch.device(config["primary_device"])
    
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
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    num_frames = len(dataset)
    
    color, depth, intrinsics, c2w = dataset[0]
    config['cameras']['height'] = color.shape[0]
    config['cameras']['width'] = color.shape[1]
    config['cameras']['intrinsics'] = intrinsics[:3, :3].detach().cpu().numpy().tolist()
    map_every = config['frontend']['map_every']
    keyframe_every = config['frontend']['keyframe_every']
    num_tracking_iters = config['frontend']['num_tracking_iters']
    num_mapping_iters = config['frontend']['num_mapping_iters']
    num_overlap_frames = config['frontend']['num_overlap_frames']
    gaussians = Gaussians(config)
    
    pcd, initial_scale = get_pointcloud(color / 255, 
                                        depth, 
                                        intrinsics, 
                                        w2c=torch.eye(4).cuda().float(), 
                                        compute_mean_sq_dist=True,
                                        sample_num=None)
    
    gaussians.create_from_pcd(pcd, initial_scale)
    
    differ_rec = []
    numpts_rec = []
    ape_rec = []
    totalpts_rec = []
    
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    total_time = 0
    frames = []
    keyframes_id = []
    keyframes_desc = []
    
    desc = GlobalDesc()
    for time_idx in tqdm(range(num_frames)):
        total_time_start = time.time()
        gt_color, gt_depth, intrinsics, gt_pose = dataset[time_idx]
        gt_color = gt_color / 255
        cur_frame = Frame(config, time_idx, gt_color, gt_depth, gt_pose, kfid=0)
        
        if time_idx < 2:
            initial_vel = torch.eye(4).cuda().float() 
        else:
            initial_vel = frames[-1].get_w2c.detach() @ torch.linalg.inv(frames[-2].get_w2c.detach()) @ frames[-1].get_w2c.detach()
        
        frames.append(cur_frame)
        cur_frame.start_optimizer(initial_transformation=initial_vel, 
                                  lr_dict=config['cameras']['frontend_lr'])
        if time_idx > 0:
            tracking_frame_start = time.time()
            for iter in tqdm(range(num_tracking_iters), total=num_tracking_iters, desc="Tracking: "):
                tracking_start_time = time.time()
                cur_frame.transform.optimizer.zero_grad(set_to_none=True)
                
                render_pkg = Renderer_tracking(config, gaussians, cur_frame)
                loss = get_loss(config, render_pkg, cur_frame, tracking=True)
                loss.backward()
                with torch.no_grad():
                    cur_frame.transform.optimizer.step()
                    cur_frame.transform.update_learning_rate()
                    
                tracking_end_time = time.time()
                tracking_iter_time_sum += tracking_end_time - tracking_start_time
                tracking_iter_time_count += 1
            
            tracking_frame_end = time.time()
            tracking_frame_time_sum += tracking_frame_end - tracking_frame_start 
            tracking_frame_time_count += 1
            
                
        if time_idx % map_every == 0:
            add_new_gaussians(config, gaussians, cur_frame)
            selected_frame_time_idx = [time_idx]
            if len(keyframes_id) > 0:
                query_desc = desc(cur_frame.gt_color.permute(2, 0, 1).unsqueeze(0))
                map_descs = torch.cat(keyframes_desc, dim=0)
                i, d = map_descs.shape
                sims = torch.einsum("id,jd->ij", map_descs, query_desc).view(i, -1)
                max_sims, _ = torch.max(sims, dim=1)
                
                _, max_sim_lmidxs = max_sims.topk(min(num_overlap_frames-1, i))
                
                selected_frame_time_idx = selected_frame_time_idx + [keyframes_id[i] for i in max_sim_lmidxs.tolist()]
            
            mapping_frame_start = time.time()
            for iter in tqdm(range(num_mapping_iters), total=num_mapping_iters, desc="Mapping: "):
                mapping_start_time = time.time()
                selected_camera_idx = random.choice(selected_frame_time_idx)
                selected_camera: Frame = frames[selected_camera_idx]
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                render_pkg = Renderer_mapping(config, gaussians, selected_camera)
                loss = get_loss(config, render_pkg, selected_camera)
                loss.backward()
                with torch.no_grad():
                    gaussians.optimizer.step()
                mapping_end_time = time.time()
                mapping_iter_time_sum += mapping_end_time - mapping_start_time
                mapping_iter_time_count += 1
            
        mapping_frame_end = time.time()
        mapping_frame_time_sum += mapping_frame_end - mapping_frame_start
        mapping_frame_time_count += 1
        if time_idx % keyframe_every == 0:
            cur_frame.finish_optimizer(save=True)
            keyframes_id.append(time_idx)
            keyframes_desc.append(desc(cur_frame.gt_color.permute(2, 0, 1).unsqueeze(0)))
        else:
            cur_frame.finish_optimizer()
        
        torch.cuda.empty_cache()
        total_time_end = time.time()
        total_time += total_time_end - total_time_start
        
        if time_idx % 10 == 0:
            w2cs = [frame.get_w2c for frame in frames]
            gt_w2cs = [frame.gt_w2c for frame in frames]
            # notbad = [id for id in range(len(gt_w2cs)) if not (torch.isnan(gt_w2cs[id]).any() or torch.isinf(gt_w2cs[id]).any())]
            # w2cs = [w2cs[id] for id in notbad] # w2cs[notbad]
            # gt_w2cs = [gt_w2cs[id] for id in notbad] #  gt_w2cs[notbad]
            vis_base_dir = config['vis_base_dir']
            os.makedirs(vis_base_dir, exist_ok=True)
            
            if len(w2cs) > 3:
                pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in w2cs]
                gt_pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in gt_w2cs]
                traj_ref = PosePath3D(poses_se3=gt_pose_w2cs)
                traj_est = PosePath3D(poses_se3=pose_w2cs)
                traj_est_aligned = copy.deepcopy(traj_est)
                traj_est_aligned.align(traj_ref, correct_scale=False,)
                pose_relation = metrics.PoseRelation.translation_part
                data = (traj_ref, traj_est_aligned)
                ape_metric = metrics.APE(pose_relation)
                ape_metric.process_data(data)
                ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
                ape_stats = ape_metric.get_all_statistics()
                
                fig = plt.figure()
                plot_mode = evo.tools.plot.PlotMode.xy
                ax = evo.tools.plot.prepare_axis(fig, plot_mode)
                ax.set_title(f"ATE RMSE: {ape_stat}")
                ape_rec.append(ape_stat)
                evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
                evo.tools.plot.traj_colormap(
                    ax,
                    traj_est_aligned,
                    ape_metric.error,
                    plot_mode,
                    min_map=ape_stats["min"],
                    max_map=ape_stats["max"],
                )            
                ax.legend()
                plt.savefig(os.path.join(f"{vis_base_dir}", "evo_2dplot.png"), dpi=90)
                plt.close()
            
            differ_rec = []
            for j in range(len(w2cs)):
                differ_rec.append(torch.norm((w2cs[j] @ torch.linalg.inv(gt_w2cs[j]))[:3, 3]).item())
            plt.plot(range(len(differ_rec)), differ_rec)
            plt.savefig(f"{vis_base_dir}/trackloss.png")
            plt.close()
            
            plt.plot(range(len(ape_rec)), ape_rec)
            plt.savefig(f"{vis_base_dir}/ape.png")
            plt.close()
                    
            num_pts = gaussians.size
            numpts_rec.append(num_pts)
            plt.plot(range(len(numpts_rec)), numpts_rec)
            plt.savefig(f"{vis_base_dir}/numpts.png")
            plt.close()
    
    # final refinement   
    num_final_refine_iters = num_frames 
    for iter in tqdm(range(num_final_refine_iters), total=num_final_refine_iters, desc="final refinement: "):
        selected_camera_idx = random.choice(keyframes_id)
        selected_camera: Frame = frames[selected_camera_idx]
        gaussians.optimizer.zero_grad(set_to_none=True)
        
        render_pkg = Renderer_mapping(config, gaussians, selected_camera)
        loss = get_loss(config, render_pkg, selected_camera)
        loss.backward()
        with torch.no_grad():
            gaussians.optimizer.step()

    num_frames = tracking_frame_time_count + 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / num_frames
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / num_frames
    avg_time = total_time / num_frames
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    print(f"Mapping time: {mapping_frame_time_count} s")
    result = {
        "tracking_iter_time(ms)": tracking_iter_time_avg * 1000,
        "tracking_frame_time(s)": tracking_frame_time_avg,
        "mapping_iter_time(ms)": mapping_iter_time_avg * 1000, 
        "mapping_frame_time(s)": mapping_frame_time_avg,
        "frame_time": avg_time, 
    }
    vis_base_dir = config['vis_base_dir'] 
    os.makedirs(vis_base_dir, exist_ok=True)
    with open(os.path.join(vis_base_dir, "time.json"), "w") as f:
        json.dump(result, f)
    w2cs = [frame.get_w2c for frame in frames]
    gt_w2cs = [frame.gt_w2c for frame in frames]
    lm = LocalMaps(w2cs, gt_w2cs)
    save_scence(config, gaussians, lm, os.path.join(vis_base_dir, "save"))
    eval_final(config, 
               gaussians, 
               w2cs,
               gt_w2cs, 
               f"{vis_base_dir}/result/")
    
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()
    
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    
    seed_everything(seed=experiment.config['seed'])
    
    rgbd_slam(experiment.config)
    
    