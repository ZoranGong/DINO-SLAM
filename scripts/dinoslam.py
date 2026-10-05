import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

import torch
import wandb
from datetime import datetime
from tqdm import tqdm
from queue import Queue

from utils.common_utils import seed_everything
from datasets import get_dataset, load_dataset_config
from slam.Frontend import Frontend
from slam.Backend import Backend
from scene import save_scence
from utils.eval import eval_final
from datasets import get_dataset, load_dataset_config
from scene import save_scence

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


def rgbd_slam(config: dict):
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if config['use_wandb']:
        wandb_run = wandb.init(
            project = config['wandb']['project_name'],
            name = f"{config['wandb']['name']}_{config['data']['sequence']}_{current_datetime}",
            config = config,
            mode = None if config['use_wandb'] else "disabled"
        )
        wandb_run.define_metric("cur_lmid") 
        wandb_run.define_metric("APE*", step_metric = "cur_lmid")
        wandb_run.define_metric("Backend_numpts", step_metric = "cur_lmid")
        wandb_run.define_metric("frame_idx")
        wandb_run.define_metric("Frontend_numpts", step_metric = "frame_idx")
    else: wandb_run = None

    vis_base_dir = config['vis_base_dir']
    dataset_config = config["data"]
    device = torch.device(config["primary_device"])

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
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    num_frames = len(dataset)
    
    color, depth, intrinsics, c2w = dataset[0]
    config['cameras']['height'] = color.shape[0]
    config['cameras']['width'] = color.shape[1]
    config['cameras']['intrinsics'] = intrinsics[:3, :3].detach().cpu().numpy().tolist()
    
    to_backend = Queue()
    frontend = Frontend(config, to_backend, wandb_run)
    backend = Backend(config, wandb_run)

    for time_idx in tqdm(range(num_frames)):
        gt_color, gt_depth, intrinsics, gt_pose = dataset[time_idx]
        
        frontend.process_frame(time_idx, gt_color / 255, gt_depth, gt_pose)
        if not to_backend.empty():
            lm = to_backend.get()
            backend.process_localmap(lm, multi_process=False)
            backend.update_vis()
            backend.update_common_visualization()

        if time_idx % 10 == 0:
            frontend.update_common_visualization()
        torch.cuda.empty_cache()
        
    frontend.process_final()
    
    while not to_backend.empty():
        lm = to_backend.get()
        backend.process_localmap(lm)
    
    backend.final_refine()
    eval_result = eval_final(config, 
                            backend.map, 
                            backend.local_maps.get_w2cs(),
                            backend.local_maps.get_gt_w2cs(), 
                            f"{vis_base_dir}/final_refine_result/")
    if config['use_wandb']:
        columns = ["Tag", "PSNR", "SSIM", "LPIPS", "Depth RMSE", "Depth L1", "ATE RMSE"]
        metrics_table = wandb.Table(columns = columns)
        metrics_table.add_data(
            "Eval Result",
            eval_result['PSNR: '],
            eval_result['SSIM: '],
            eval_result['LPIPS: '],
            eval_result['Depth RMSE: '],
            eval_result['Depth L1: '],
            eval_result['ATE RMSE: ']
        )
        wandb_run.log({"Metrics": metrics_table})
    
    save_scence(config, backend.map, backend.local_maps, f"{vis_base_dir}/save/")
    backend.save_sge(f"{vis_base_dir}/save/sge.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()
    
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()
    
    seed_everything(seed=experiment.config['seed'])
    
    rgbd_slam(experiment.config)
    