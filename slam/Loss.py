import torch
from utils.loss_utils import l1_loss
from utils.common_utils import get_pts_from_depth, get_normal_from_pts
from scene import Frame 

def weight_losses(loss_dict: dict, loss_weight: dict):
    loss = torch.tensor(0., device='cuda')
    for k, v in loss_dict.items():
        w = loss_weight.get(k, 0.0)
        if w != 0:
            loss = loss + w * v
    return loss

def get_loss(config: dict, render_pkg: dict, camera: Frame, tracking=False, extra_losses: dict = None):

    loss_dict = {}
    gt_color = camera.gt_color
    gt_depth = camera.gt_depth
    enable_normal_loss = config['loss']['use_normal_loss']
    loss_weight = config['loss']['tracking' if tracking else 'mapping']
    
    if enable_normal_loss:
        H = config['cameras']['height']
        W = config['cameras']['width']
        K = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
        gt_pts = get_pts_from_depth(H, W, K, gt_depth)
        gt_normal = get_normal_from_pts(H, W, gt_pts)
        gt_normal_mask = (gt_normal.norm(dim=-1) > 1e-5).reshape(-1)
    
    render_alpha = torch.nan_to_num(render_pkg['render_alpha'], 0, 0).permute(1, 2, 0)
    render_depth = torch.nan_to_num(render_pkg['render_depth'], 0, 0).permute(1, 2, 0)
    render_color = torch.nan_to_num(render_pkg['render_color'], 0, 0).permute(1, 2, 0)
    render_dist = torch.nan_to_num(render_pkg['render_dist'], 0, 0).permute(1, 2, 0)
    
    if enable_normal_loss:
        render_normal = torch.nan_to_num(render_pkg['render_normal'], 0, 0)
        normal_mask = (render_normal.norm(dim=0) > 1e-5).view(-1)
        render_normal = render_normal / render_normal.norm(dim=0)[None]
        render_normal = render_normal.permute(1, 2, 0)
        
    loss_dict = {}
    depth_mask = (gt_depth > 1e-5).view(-1) & (render_depth > 1e-5).view(-1) # 剔除深度坏点
    if tracking:
        tracking_depth_mask = depth_mask
        if config['loss']['ignore_outliners']:
            depth_error = l1_loss(render_depth, gt_depth).view(-1) * depth_mask
            inliner_mask = depth_error < 10 * depth_error.median()
            tracking_depth_mask = inliner_mask & tracking_depth_mask
        
        alpha_mask = (render_alpha > config['loss']['silmask_th']).view(-1)
        tracking_depth_mask = tracking_depth_mask & alpha_mask 
        tracking_color_mask = tracking_depth_mask
        
        loss_dict['color'] = l1_loss(render_color, gt_color).view(-1, 3)[tracking_color_mask].sum()
        loss_dict['depth'] = l1_loss(render_depth, gt_depth).view(-1, 1)[tracking_depth_mask].sum() # (w * l1_loss(render_depth, gt_depth))[tracking_depth_mask].sum() # fix_log_loss(sampled_depth, gt_sampled_depth, tracking_mask).sum() # 
        if enable_normal_loss:
            loss_dict['normal'] = (1 - (render_normal.view(-1, 3) * gt_normal.view(-1, 3).sum(dim=-1)[:, None]))[tracking_depth_mask & normal_mask & gt_normal_mask].sum()

    else:
        mapping_depth_mask = depth_mask
        mapping_color_mask = (render_alpha > config['densify']['edge_thres']).reshape(-1) if config['densify']['use_edge_growth'] else depth_mask
        loss_dict['color'] = (l1_loss(render_color, gt_color).view(-1, 3))[mapping_color_mask].mean()
        loss_dict['depth'] = (l1_loss(render_depth, gt_depth).view(-1, 1))[mapping_depth_mask].mean()
        if enable_normal_loss:
            loss_dict['normal'] = ((1 - (render_normal.view(-1, 3) * gt_normal.view(-1, 3)).sum(dim=-1)[:, None]))[mapping_depth_mask & gt_normal_mask & normal_mask].mean()
        loss_dict['dist'] = (render_dist.view(-1, 1)[mapping_color_mask]).mean()

    if extra_losses:
        loss_dict.update(extra_losses)

    return weight_losses(loss_dict, loss_weight)
        