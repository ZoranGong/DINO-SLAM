import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import random
import sys
import shutil
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

import time
import torch
import numpy as np
import wandb
from tqdm import tqdm
from datetime import datetime
import torch.multiprocessing as mp

from utils.common_utils import seed_everything
from datasets import get_dataset, load_dataset_config
from slam.Frontend import mp_Frontend
from slam.Backend import mp_Backend
from scene import save_scence
from utils.eval import eval_final
from datasets import get_dataset, load_dataset_config

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt


class DataFeeder:
    
    def __init__(self, config, dataflow: mp.Queue, event):
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
        self.dataset = get_dataset(
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
        
    
        color, depth, intrinsics, c2w = self.dataset[0]
        config['cameras']['height'] = color.shape[0]
        config['cameras']['width'] = color.shape[1]
        config['cameras']['intrinsics'] = intrinsics[:3, :3].detach().cpu().numpy().tolist()

        self.dataflow = dataflow
        self.event = event
        
    def run(self, ):
        num_frames = len(self.dataset)
        for time_idx in tqdm(range(num_frames), total=num_frames):
            gt_color, gt_depth, intrinsics, gt_pose = self.dataset[time_idx]
            self.dataflow.put({"data":(time_idx, gt_color / 255, gt_depth, gt_pose)})
            
            while self.dataflow.qsize() > 5:
                time.sleep(0.1)
           
        self.dataflow.put("finish")
        self.event.wait()
        print("Datafeeder process finished !!!")
                
def rgbd_slam(config: dict):
    if config['use_wandb']:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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

    mp.set_start_method('spawn', force=True)
    
    vis_base_dir = config['vis_base_dir']
    sync_event = mp.Event()
    data_flow = mp.Queue()
    to_backend = mp.Queue()
    datafeeder = DataFeeder(config, data_flow, sync_event)
    frontend = mp_Frontend(config, data_flow, to_backend, sync_event, wandb_run)
    backend = mp_Backend(config, to_backend, sync_event, wandb_run)
   
    frontend_process = mp.Process(target=frontend.run)
    datafeeder_process = mp.Process(target=datafeeder.run)
    
    frontend_process.start()
    datafeeder_process.start()
    backend.run()
    frontend_process.join()
    datafeeder_process.join()
    os.makedirs(vis_base_dir, exist_ok=True)
    
    # eval_final(config, 
    #            backend.map, 
    #            backend.local_maps.get_w2cs(),
    #            backend.local_maps.get_gt_w2cs(), 
    #            f"{vis_base_dir}/slam_result/")
    
    backend.final_refine()
        
    eval_result = eval_final(config, 
                            backend.map, 
                            backend.local_maps.get_w2cs(),
                            backend.local_maps.get_gt_w2cs(), 
                            f"{vis_base_dir}/result/")
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
    
    