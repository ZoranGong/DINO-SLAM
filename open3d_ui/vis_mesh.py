
import os
import time
import open3d as o3d 
import torch 
from datasets import load_dataset_config, get_dataset
import numpy as np 

from tqdm import tqdm 
from render import Renderer_view
import multiprocessing
import threading
import queue
import matplotlib.pyplot as plt 
import matplotlib
from utils.common_utils import get_pts_from_depth
import random

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), 
                    center=cylinder_segment.get_center())
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

class Ui_o3d:
    
    def __init__(self, config, first_view):
        self.config = config 
        self.first_view = first_view
        self.queue = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.run, args=())
        self.p.start()
    
    def destroy(self, ):
        self.queue.put(("finish", ))
        self.p.join()   
        
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
        
    def run(self):
        task_queue = self.queue
        config = self.config
        self.min_error = 0
        self.max_error = 0.01
        self.viz_scale = config['viz']['view_scale']
        self.video_freq = config['viz']['video_freq']
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
        

        if 'cam_loc' in config.get('viz', {}):
            extrinsic=np.array(config['viz']['cam_loc'])
        else:
            extrinsic = self.first_view
            extrinsic[:3, 3] = extrinsic[:3, 3] #  + np.array([0, 0, 0.1])

        # Open3d bug, need a point in vision
        Rc2w = extrinsic[:3, :3].T  # 3x3
        Tc2w = -extrinsic[:3, :3] @ extrinsic[:3, 3:] # 3x1
        insight_point_c = np.array([0, 0, 0]).T
        insight_point_w =  Rc2w @ insight_point_c + Tc2w
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(insight_point_w)
        self.pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(insight_point_w))
        self.vis.add_geometry(self.pcd)
        
        self.gen_animation = self.config['viz']['gen_animation']
        self.viz_dir = os.path.join(self.config['vis_base_dir'], "open3d")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        ctr = self.vis.get_view_control()
        ctr.set_constant_z_near(0.001)
        ctr.set_constant_z_far(1000)
        cparams = o3d.camera.PinholeCameraParameters()
        cparams.extrinsic = extrinsic
        cparams.intrinsic.intrinsic_matrix = self.intrinsics
        cparams.intrinsic.height = self.viz_h
        cparams.intrinsic.width = self.viz_w
        ctr.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
        
        opt = self.vis.get_render_option()
        opt.point_size = 4
        opt.line_width = 10
        opt.mesh_show_back_face = False
        
        self.mesh = None 
        self.error_list = []
        self.w2c_list = []
        self.pre_frustums = []
        self.pre_lines = []
        self.pts = None 

        prtc_id = 0
        while True:
            if not task_queue.empty():
                cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
                data = task_queue.get()
                if data[0] == "finish":
                    break 
                
                if data[0] == "mesh":
                    if self.mesh is not None:
                        self.vis.remove_geometry(self.mesh)
                    
                    vert, tria, colors = data[1:]
                    self.mesh = o3d.geometry.TriangleMesh()
                    self.mesh.vertices = o3d.utility.Vector3dVector(vert) 
                    self.mesh.triangles = o3d.utility.Vector3iVector(tria)  
                    self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors) 
                    self.vis.add_geometry(self.mesh)
                
                if data[0] == "camera":
                    cam_w2c, error = data[1:]
                    self.w2c_list.append(cam_w2c)
                    self.error_list.append(error)
                    
                    if len(self.w2c_list) > 2:
                        for frustum in self.pre_frustums:
                            self.vis.remove_geometry(frustum)
                        self.pre_frustums = []
                        for line in self.pre_lines:
                            self.vis.remove_geometry(line)
                        self.pre_lines = []
                        
                        frustum_size = 0.6
                        num_t = len(self.w2c_list)
                        cam_centers = []
                        cam_colormap = plt.get_cmap('cool')
                        norm_factor = 0.5
                        h = self.config['viz']['viz_h']
                        w = self.config['viz']['viz_w']
                        k = self.intrinsics
                        # current_camera_parameters = self.view_control.convert_to_pinhole_camera_parameters()

                        for i_t in range(num_t):
                            frustum_color = (0.5, 0.95, 1.0)#cam_colormap(i_t * norm_factor / num_t)[:3]
                            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, self.w2c_list[i_t], frustum_size)
                            frustum.paint_uniform_color(np.array(frustum_color))
                            if i_t == num_t -1 :
                                self.vis.add_geometry(frustum)
                            cam_centers.append(np.linalg.inv(self.w2c_list[i_t])[:3, 3])
                            self.pre_frustums.append(frustum)


                        # Initialize Camera Trajectory
                        num_lines = [1]
                        total_num_lines = num_t - 1
                        cols = []
                        line_colormap = plt.get_cmap('jet')
                        norm_factor = 0.5
                        if total_num_lines > 0:
                            for line_t in range(total_num_lines):
                                self.max_error = max(self.error_list)
                                cols.append(np.array(line_colormap((self.error_list[line_t]) / (self.max_error))[:3]))
                            cols = np.array(cols)
                            all_cols = [cols]
                            out_pts = [np.array(cam_centers)]
                            linesets = self.make_lineset(out_pts, all_cols, num_lines)
                            lines = o3d.geometry.LineSet()
                            lines.points = linesets[0].points
                            lines.colors = linesets[0].colors
                            lines.lines = linesets[0].lines

                            # self.vis.add_geometry(lines)
                            linemesh = LineMesh(points = cam_centers, colors = cols, radius = 0.025)
                            linemesh.add_line(self.vis)

                            self.pre_lines.append(lines)
                    
                if data[0] == "points":
                    if self.pts is not None:
                        self.vis.remove_geometry(self.pts)
                    self.pts = o3d.geometry.PointCloud()
                    self.pts.points = o3d.utility.Vector3dVector(data[1])
                    
                    colors = np.zeros_like(data[1])
                    colors[:, 0] = 0.5
                    colors[:, 1] = 0.95
                    colors[:, 2] = 1.0
                    self.pts.colors = o3d.utility.Vector3dVector(colors)
                    self.vis.add_geometry(self.pts)
                        
                cam = self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
                
                # view_control = self.vis.get_view_control()
                # print(view_control.convert_to_pinhole_camera_parameters().extrinsic)
                
                if self.gen_animation:
                    img = np.asarray(self.vis.capture_screen_float_buffer(True))
                    # plt.figure(figsize=(3, 3))
                    img_height, img_width = img.shape[:2]
                    img_aspect = (img_width) / float(img_height)
                    fig_width = 5 #8
                    fig_height = fig_width / (img_aspect)
                    fig = plt.figure(figsize=(fig_width, fig_height))
                    ax = plt.gca()
                    ax.imshow(img, extent=[0, img_width, 0, img_height])
                    ax.axis('off')
                    
                    cmap = plt.cm.jet
                    # norm = matplotlib.colors.Normalize(vmin=0, vmax=0.001)
                    norm = matplotlib.colors.Normalize(vmin=self.min_error, vmax=self.max_error)
                    mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
                    mappable.set_array([])
                    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.01, shrink=0.6, ticks=[])
                    cbar.set_label(f"0 cm ~ {round(self.max_error * 100, 2)} cm")
                    fig.tight_layout()
                    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

                    plt.savefig(os.path.join(self.viz_dir, f"{prtc_id:05d}.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                    prtc_id += 1
                    
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.1)
        
        if self.gen_animation:
            os.system(f"/usr/bin/ffmpeg -f image2 -r {self.video_freq} -pattern_type glob -i '{self.viz_dir}/*.png' -y {self.viz_dir}/../mesh.mp4")
            os.system(f"/usr/bin/ffmpeg -i {self.viz_dir}/../mesh.mp4 -c:v libx264 -crf 18 {self.viz_dir}/../mesh_compressed.mp4")
        
class Vis_Mesh:
    def __init__(self, config, ):
        
        self.scale = 1.0
        self.mesh_freq = config['viz']['mesh_every']
        self.config = config
        self.first_w2c = self.get_first_w2c()
        self.ui = Ui_o3d(self.config, self.first_w2c)
        
        self.queue = queue.Queue()
        self.proc = threading.Thread(target=self.run, args=())
        self.proc.start()
    
    def destroy(self, ):
        self.ui.destroy()
        self.queue.put((None, None, None, None, None))
        self.proc.join()
    
    def update_frame(self, color, depth, w2c, gt_w2c, time_idx):
        color_np = torch.clamp(color.permute(1, 2, 0), 0., 1.).contiguous().detach().cpu().numpy() * 255
        color_np = color_np.astype(np.uint8)
        depth_np = depth.squeeze(0).detach().cpu().numpy()
        
        w2c_np = w2c.detach().cpu().numpy()
        gt_w2c_np = gt_w2c.detach().cpu().numpy()
        
        self.queue.put((color_np, depth_np, w2c_np, gt_w2c_np, time_idx))
        
    def run(self, ):
        intrinsics_list = self.config['cameras']['intrinsics']
        self.width = self.config['cameras']['width']
        self.height = self.config['cameras']['height']
        
        self.intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, intrinsics_list[0][0], intrinsics_list[1][1], intrinsics_list[0][2], intrinsics_list[1][2]         
        )
        self.intrinsics_np = np.array(intrinsics_list)
        
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=5.0 * self.scale / 512.0,
            sdf_trunc=0.04 * self.scale,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        self.time_idx = 0
        while True:
            time.sleep(0.1)
            if not self.queue.empty():
                color_np, depth_np, w2c, gt_w2c, time_idx = self.queue.get()
                if color_np is None:
                    break 
                self.time_idx = time_idx
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image( np.ascontiguousarray(color_np)),
                o3d.geometry.Image(depth_np),
                depth_scale=self.scale,
                depth_trunc=30,
                convert_rgb_to_intensity=False)
                self.volume.integrate(rgbd, self.intrinsics_o3d,  w2c @ self.first_w2c)
                
                # pose 
                cam_w2c = w2c @ self.first_w2c
                if (np.isnan(gt_w2c)).any():
                    error = 0
                else:
                    error = np.linalg.norm((w2c @ np.linalg.inv(gt_w2c))[:3, 3])
                
                self.ui.queue.put(("camera", cam_w2c, error))
                
                # points
                intrinsics = self.config['cameras']['intrinsics']
                CX, CY, FX, FY = intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1]
                x_grid, y_grid = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing="xy")
                xx = (x_grid - CX) / FX 
                yy = (y_grid - CY) / FY
                xx = xx.reshape(-1)
                yy = yy.reshape(-1)
                depth_z = depth_np.reshape(-1)
                pts_c = np.stack((xx * depth_z, yy * depth_z, depth_z, np.ones_like(depth_z)), axis=-1)
                sample_idxs = random.sample(range(len(pts_c)), (self.width * self.height) // 100)
                pts_c = pts_c[sample_idxs]
                pts_w = pts_c @ np.linalg.inv(cam_w2c).T
                self.ui.queue.put(("points", pts_w[:, :3]))
                
                if time_idx % self.mesh_freq == 0:
                    o3d_mesh = self.volume.extract_triangle_mesh()
                    compensate_vector = (-0.0 * self.scale / 512.0, 2.5 *
                                            self.scale / 512.0, -2.5 * self.scale / 512.0)
                    o3d_mesh = o3d_mesh.translate(compensate_vector)
                    vert = np.asarray(o3d_mesh.vertices)
                    tria = np.asarray(o3d_mesh.triangles)
                    colors = np.asarray(o3d_mesh.vertex_colors)
                    self.ui.queue.put(("mesh", vert, tria, colors))
                
    
    def get_first_w2c(self, ) -> np.ndarray:
        config = self.config
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
        _, _, _, first_c2w = dataset[0]
        first_w2c = torch.linalg.inv(first_c2w).detach().cpu().numpy()
        return first_w2c