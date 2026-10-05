import sys
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

from scene import Gaussians, Frame
from render import Renderer_view
import torch
from utils.common_utils import get_pointcloud
import numpy as np
import time
from scene import load_scence, Gaussians, Frame
import argparse
import time
import os 
import open3d as o3d
import matplotlib.pyplot as plt 



class Visualization:
    
    def __init__(self, config: dict) -> None:
        
        self.config = config
        self.viz_scale = config['viz']['view_scale']
        config['viz']['viz_h'] = 340
        config['viz']['viz_w'] = 600
        self.viz_h = int(config['viz']['viz_h'] * self.viz_scale)
        self.viz_w = int(config['viz']['viz_w'] * self.viz_scale)
        
        self.ori_h = config['cameras']['height']
        self.ori_w = config['cameras']['width']
        self.intrinsics = np.array(config['cameras']['intrinsics'], dtype=np.float32)[:3, :3] * self.viz_scale
        
        self.intrinsics[0, :] *= config['viz']['viz_w'] / self.ori_w
        self.intrinsics[1, :] *= config['viz']['viz_h'] / self.ori_h
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width   = self.viz_w, 
                               height  = self.viz_h,
                               visible = True)
        self.is_created = False
    
    def create_visualization(self, gaussians: Gaussians, first_w2c):
        extrinsic = first_w2c.detach().contiguous().double().cpu().numpy()
        extrinsic[:3, 3] = extrinsic[:3, 3] + np.array([0, 0, 0.1])
        
        Rc2w = extrinsic[:3, :3].T  # 3x3
        Tc2w = -extrinsic[:3, :3] @ extrinsic[:3, 3:] # 3x1
        insight_point_c = np.array([0, 0, 0]).T
        insight_point_w =  Rc2w @ insight_point_c + Tc2w
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(insight_point_w)
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(insight_point_w))
        self.vis.add_geometry(self.pcd)
        
        self.pcd = o3d.geometry.PointCloud()
        with torch.no_grad():
            render_pkg = Renderer_view(self.config, gaussians, w2c=first_w2c)
            color = render_pkg['render_color'].permute(1, 2, 0)
            depth = render_pkg['render_depth'].permute(1, 2, 0)
            depth = torch.nan_to_num(depth / render_pkg['render_alpha'].permute(1, 2, 0), 0, 0)
            point_cloud = get_pointcloud(color, depth, self.config['cameras']['intrinsics'], w2c=first_w2c)
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud.points.contiguous().double().cpu().numpy())
        self.pcd.colors = o3d.utility.Vector3dVector(point_cloud.colors.contiguous().double().cpu().numpy())
        self.vis.add_geometry(self.pcd)
        
        self.view_control = self.vis.get_view_control()
        cparams = o3d.camera.PinholeCameraParameters()
        cparams.extrinsic = extrinsic
        cparams.intrinsic.intrinsic_matrix = self.intrinsics
        cparams.intrinsic.height = self.viz_h
        cparams.intrinsic.width = self.viz_w
        self.view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

        render_options = self.vis.get_render_option()
        render_options.point_size = self.viz_scale
        render_options.light_on = False
        render_options.show_coordinate_frame = False
        self.pre_frustums = []
        self.pre_lines = []
        
        self.is_created = True
        
    def update(self, gaussians: Gaussians):
        if self.vis is None: return
        cam_params = self.view_control.convert_to_pinhole_camera_parameters()

        w2c = torch.tensor(cam_params.extrinsic).cuda().float()
        with torch.no_grad():
            ts = time.time()
            render_pkg = Renderer_view(self.config, gaussians, w2c = w2c)
            td = time.time()
            color = render_pkg['render_color'].permute(1, 2, 0)
            depth = render_pkg['render_depth'].permute(1, 2, 0)
            
            depth = torch.nan_to_num(depth / render_pkg['render_alpha'].permute(1, 2, 0), 0, 0)
            
            point_cloud = get_pointcloud(color, depth, self.config['cameras']['intrinsics'], w2c=w2c)
            
            if point_cloud.points.shape[0] > 0:
                self.pcd.points = o3d.utility.Vector3dVector(point_cloud.points.contiguous().double().cpu().numpy())
                self.pcd.colors = o3d.utility.Vector3dVector(point_cloud.colors.contiguous().double().cpu().numpy())
            
                self.vis.update_geometry(self.pcd)
                
        if not self.vis.poll_events():
            self.destory()
            return 
        self.vis.update_renderer()
        return td - ts
            
    def destory(self,):
        self.vis.destroy_window()
        del self.view_control
        self.view_control = None
        del self.vis 
        self.vis = None
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Path to experiment file")
    args = parser.parse_args()


    result_path = os.path.join(args.m, "save")
    config, gaussians, w2cs, gt_w2cs = load_scence(result_path)
    
    vis = Visualization(config)
    vis.create_visualization(gaussians, w2cs[0])
    
    total_time = 0
    cnt = 0
    while True:
        time.sleep(0.1)
        total_time += vis.update(gaussians)
        cnt = cnt + 1
        if cnt % 100 == 0:
            print("fps:", 1 / (total_time / cnt))