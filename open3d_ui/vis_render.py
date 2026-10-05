import open3d as o3d
from scene import Gaussians, Frame
from render import Renderer_view
import torch
from utils.common_utils import get_pointcloud
import numpy as np
import matplotlib.pyplot as plt 
import time
import cv2
import os 
import copy 
import threading
import queue 


class Vis_Render:
    
    def __init__(self, config: dict, save_dir) -> None:
        
        self.config = config
        self.viz_scale = config['viz']['view_scale']
        self.video_freq = config['viz']['video_freq']
        self.viz_h = int(config['viz']['viz_h'] * self.viz_scale)
        self.viz_w = int(config['viz']['viz_w'] * self.viz_scale)
        
        self.ori_h = config['cameras']['height']
        self.ori_w = config['cameras']['width']
        self.intrinsics = np.array(config['cameras']['intrinsics'], dtype=np.float32)[:3, :3] * self.viz_scale
        
        self.intrinsics[0, :] *= config['viz']['viz_w'] / self.ori_w
        self.intrinsics[1, :] *= config['viz']['viz_h'] / self.ori_h
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.offset = (0, 0, 0.1) #0.28
        self.first_w2c = np.identity(4)
        self.gen_animation = True 
        self.queue = queue.Queue()
        self.proc = threading.Thread(target=self.run, args=())
        self.proc.start()

    def run(self):
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width   = self.viz_w, 
                               height  = self.viz_h,
                               visible = True)
        
        # 设置初始视角沿z轴向后偏移0.5
        extrinsic = self.first_w2c
        extrinsic[:3, 3] = extrinsic[:3, 3] + self.offset
        
        Rc2w = extrinsic[:3, :3].T  # 3x3
        Tc2w = -extrinsic[:3, :3] @ extrinsic[:3, 3:] # 3x1
        insight_point_c = np.array([0, 0, 0]).T
        insight_point_w =  Rc2w @ insight_point_c + Tc2w
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(insight_point_w)
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(insight_point_w))
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
        self.cam_w2cs = []
        self.cam_types = []
        self.type_info = [
            (0.95, 0.6, 0.0, 0.045), 
            (0.  , 1. , 0. , 0.025),
            (0.  , 0. , 1. , 0.025),
            [1.  , 0. , 0. , 0.045]]           
        self.cnt = 0
        
        while True:
            time.sleep(0.5)
            
            if not self.queue.empty():
                w2c, type, points, colors, time_idx = self.queue.get()
                
                if w2c is None:
                    break 
                
                self.cam_w2cs.append(w2c)
                self.cam_types.append(type)
                
                self.draw_cams()
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self.vis.update_geometry(self.pcd)
                self.vis.capture_screen_image(os.path.join(self.save_dir, f"frame_{time_idx:05d}.png"))
                
            self.vis.poll_events()
            self.vis.update_renderer()

        if self.gen_animation:
            os.system(f"/usr/bin/ffmpeg -f image2 -r {self.video_freq} -pattern_type glob -i '{self.save_dir}/*.png' -y {self.save_dir}/../render.mp4")
            os.system(f"/usr/bin/ffmpeg -i {self.save_dir}/../render.mp4 -c:v libx264 -crf 18 {self.save_dir}/../render_compressed.mp4")
        self.vis.destroy_window()
    
    def make_lineset(self, all_pts, all_cols, num_lines):
        linesets = []
        for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
            lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
            pt_indices = np.arange(len(lineset.points))
            line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
            lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
            linesets.append(lineset)
        return linesets

    def draw_cams(self, ):
        current_camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()
        all_w2cs = self.cam_w2cs
        for frustum in self.pre_frustums:
            self.vis.remove_geometry(frustum)
        self.pre_frustums = []
        for line in self.pre_lines:
            self.vis.remove_geometry(line)
        self.pre_lines = []
        
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        h = self.config['viz']['viz_h']
        w = self.config['viz']['viz_w']
        k = self.intrinsics
        # current_camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()
        for i_t in range(num_t):
            
            t = self.cam_types[i_t] if i_t < num_t - 1 else -1
            frustum_size  = self.type_info[t][-1]
            frustum_color = self.type_info[t][:3]
            
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(frustum_color))

            frustum_color = cam_colormap(i_t * norm_factor / num_t)[:3]
            self.vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])
            self.pre_frustums.append(frustum)

        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        if total_num_lines > 0:
            for line_t in range(total_num_lines):
                cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = self.make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            self.vis.add_geometry(lines)
            self.pre_lines.append(lines)
            
        extrinsic = copy.deepcopy(all_w2cs[-1])
        extrinsic[:3, 3] = extrinsic[:3, 3] + self.offset
        current_camera_parameters.extrinsic = extrinsic
        self.view_control.convert_from_pinhole_camera_parameters(current_camera_parameters)


    def update_frame(self, gaussians: Gaussians, cam_w2c: torch.Tensor, cam_type, time_idx):
        render_w2c = cam_w2c.clone()
        render_w2c[:3, 3] += torch.Tensor(self.offset).cuda().float()
        
        with torch.no_grad():
            render_pkg = Renderer_view(self.config, gaussians, w2c = render_w2c)
            color = render_pkg['render_color'].permute(1, 2, 0)
            depth = render_pkg['render_depth'].permute(1, 2, 0)
            point_cloud = get_pointcloud(color, depth, self.config['cameras']['intrinsics'], w2c=render_w2c)
            points = point_cloud.points.contiguous().double().cpu().numpy()
            colors = point_cloud.colors.contiguous().double().cpu().numpy()
        
        cam_w2c = cam_w2c.detach().contiguous().double().cpu().numpy()
        self.queue.put((cam_w2c, cam_type, points, colors, time_idx))

    def destory(self, ):
        self.queue.put((None, None, None, None, None))
        self.proc.join()
        
    def reset(self, ):
        self.cam_w2cs = []
        self.cam_types = []
        