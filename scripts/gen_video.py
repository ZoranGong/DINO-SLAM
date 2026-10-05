
import sys
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

import argparse
import torch 
from tqdm import tqdm 

from scene import load_scence
from datasets import load_dataset_config, get_dataset
from render import Renderer_view
from open3d_ui import Vis_Mesh
from evo.core.trajectory import PosePath3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Path to experiment file")
    args = parser.parse_args()

    result_path = os.path.join(args.m, "save")
    config, gaussians, w2cs, gt_w2cs = load_scence(result_path)
    
    pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in w2cs]
    gt_pose_w2cs = [torch.linalg.inv(w2c).detach().cpu().numpy() for w2c in gt_w2cs]
    traj_gt = PosePath3D(poses_se3=gt_pose_w2cs)
    traj_est = PosePath3D(poses_se3=pose_w2cs)
    traj_gt.align(traj_est, correct_scale=False,)
    gt_w2cs = [torch.linalg.inv(torch.from_numpy(c2w).cuda()) for c2w in traj_gt.poses_se3]

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
    
    config['viz']['mesh_every'] = 5
    config['viz']['gen_animation'] = True
    config['viz']['video_freq'] = 15
    # scannetpp b20
    config['viz']['cam_loc'] = ([[ -0.02, -1.0,  0.0,-3.5],
                                 [-1.0, 0.02, -0.0, 2.95],
                                 [ 0.0, -0.0, -1.0, 6.94],
                                 [ 0.0,  0.0,  0.0, 1.0 ]])   

    viz = Vis_Mesh(config)

    num_frames = len(dataset)
    for time_idx in tqdm(range(num_frames), total=num_frames):
        
        gt_color, gt_depth, intrinsic, gt_pose = dataset[time_idx]
        pred_w2c = w2cs[time_idx]
        gt_w2c = gt_w2cs[time_idx]
        render_pkg = Renderer_view(config, gaussians, pred_w2c)
        
        render_color = torch.clamp(render_pkg['render_color'], 0.0, 1.0)
        render_depth = render_pkg['render_depth']
        
        if time_idx % 5 == 0:
            viz.update_frame(render_color, render_depth, pred_w2c, gt_w2c, time_idx)
    
    viz.destroy()
    
    




    