
from scene import Gaussians, Frame
import torch 
from render import Renderer_view

from utils.common_utils import get_pointcloud

def add_new_gaussians(config: dict, gaussian: Gaussians, camera: Frame, render_pkg = None, fix_w2c=None, feat_map=None):
    with torch.no_grad():
        if render_pkg is None: 
            render_pkg = Renderer_view(config, gaussian, w2c=camera.get_w2c if fix_w2c is None else fix_w2c)
        depth = render_pkg['render_depth'].permute(1, 2, 0)
        presence_sil_mask = render_pkg['render_alpha'].squeeze()
        depth = torch.nan_to_num(depth, 0, 0)
        
        if config['densify']['method'] == 'splatam':
            sil_mask = presence_sil_mask < config['densify']['sil_thres']
            depth_error = ((camera.gt_depth > 0) * torch.abs(depth - camera.gt_depth)).squeeze()
            add_mask = torch.logical_or(sil_mask, (depth > camera.gt_depth).squeeze() * (depth_error > 50 * depth_error.median()))
            pts, initial_scale = get_pointcloud(camera.gt_color, 
                                 camera.gt_depth, 
                                 camera.intrinsics, 
                                 w2c=camera.get_w2c if fix_w2c is None else fix_w2c,
                                 color_mask=add_mask,
                                 compute_mean_sq_dist=True,
                                 sample_num=config['densify']['num_addpts'],
                                 feat=feat_map)
            gaussian.add_gaussians_from_pcd(pts, initial_scale)

        if config['densify']['use_edge_growth']:
            add_mask = torch.logical_and(presence_sil_mask > config['densify']['edge_thres'], presence_sil_mask < config['densify']['sil_thres'])
            add_mask = torch.logical_and(add_mask, (camera.gt_depth < 0.001).squeeze())
            pts, initial_scale = get_pointcloud(camera.gt_color, 
                                 depth, 
                                 camera.intrinsics, 
                                 w2c=camera.get_w2c if fix_w2c is None else fix_w2c,
                                 color_mask=add_mask,
                                 compute_mean_sq_dist=True,
                                 sample_num=config['densify']['num_addpts'],
                                 feat=feat_map)
            gaussian.add_gaussians_from_pcd(pts, initial_scale)

    prune_gaussians(config, gaussian)

def prune_gaussians(config, gaussian: Gaussians):
    opacity = gaussian.get_opacity[:, 0]
    scaling = gaussian.get_scaling.mean(dim=-1)
    prune_mask = torch.logical_or(opacity < config['densify']['opacity_cuil'],
                                  scaling < config['densify']['scale_cuil'])
    
    prune_mask = torch.logical_or(prune_mask, scaling > config['densify']['scale_max'])
    gaussian.remove_gaussians_from_mask(prune_mask)