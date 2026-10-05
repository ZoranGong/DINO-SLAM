from . import render_2dgs
from . import render_3dgs
from scene import Frame, Gaussians
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

setup_camera = {
    '2dgs': render_2dgs.setup_camera,
    '3dgs': render_3dgs.setup_camera,
}

render = {
    '2dgs': render_2dgs.render,
    '3dgs': render_3dgs.render,
}

def Renderer_tracking(config: dict, gaussians:Gaussians, camera:Frame = None, fix_w2c=None, fix_exposure=None):
    method = config['render']['method']
    h = config['cameras']['height']
    w = config['cameras']['width']
    intrinsics = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
    
    view_offset = torch.eye(4, dtype=torch.float32, device='cuda')
    if method == '2dgs' and not config['render']['use_sa']:
        cam = setup_camera[method](w, h, intrinsics, view_offset, use_sa=False)
    else: 
        cam = setup_camera[method](w, h, intrinsics, view_offset)
    
    params = gaussians.get_render_params
    params.pop('features', None)
    w2c = camera.get_w2c if fix_w2c is None else fix_w2c
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params[k] = v.detach()

    params['means3D'] = (w2c[:3, :3] @ params['means3D'].T  + w2c[:3, 3:]).T
    params['rotations'] = quaternion_multiply(matrix_to_quaternion(w2c[None, :3, :3]), params['rotations']).detach()
    points2D = torch.zeros_like(gaussians.get_xyz, requires_grad=True)
    points2D.retain_grad()
    params['means2D'] = points2D
    render_pkg = render[method](cam, **params)
    
    if config['render']['enable_exposure']:
        exposure = (camera.exposure._exposure if fix_exposure is None else fix_exposure).detach()
        render_pkg['render_color'] = exposure[0] * render_pkg['render_color'] + exposure[1]
        
    if config['render']['use_weight_norm']:
        render_pkg['render_depth'] = render_pkg['render_depth'] / (render_pkg['render_alpha'] + config['render']['eps'])
        outliner_mask = torch.logical_or(render_pkg['render_depth'] > config['render']['depth_far'], render_pkg['render_depth'] < config['render']['depth_near']) 
        render_pkg['render_depth'][outliner_mask] = 0    
    return render_pkg

def feature_render_args(config: dict):
    sge = config.get('sge', {})
    return dict(feat_checkpoint=sge.get('feat_render_checkpoint', False),
                feat_scale=float(sge.get('feat_render_scale', 0.25)),
                feat_detach_geometry=not sge.get('feat_grad_to_geometry', False))

def Renderer_mapping(config: dict, gaussians:Gaussians, camera:Frame = None, fix_w2c=None, fix_exposure=None, render_feat=False):
    method = config['render']['method']
    h = config['cameras']['height']
    w = config['cameras']['width']
    intrinsics = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
    view_offset = torch.eye(4, dtype=torch.float32, device='cuda')
    
    w2c = (camera.get_w2c if fix_w2c is None else fix_w2c).detach()
    cam = setup_camera[method](w, h, intrinsics, view_offset @ w2c)
    
    if method == '2dgs' and not config['render']['use_sa']:
        cam = setup_camera[method](w, h, intrinsics, view_offset @ w2c, use_sa=False)
    else: 
        cam = setup_camera[method](w, h, intrinsics, view_offset @ w2c)

    params = gaussians.get_render_params
    if not render_feat or params.get('features', None) is None:
        params.pop('features', None)
    else:
        params.update(feature_render_args(config))
    points2D = torch.zeros_like(gaussians.get_xyz, requires_grad=True)
    points2D.retain_grad()
    params['means2D'] = points2D
    render_pkg = render[method](cam, **params)

    if config['loss']['use_normal_loss']:
        render_pkg["render_normal"] = torch.mm(w2c[:3, :3], render_pkg["render_normal"].view(3, -1)).view(3, w, h)
    if config['render']['enable_exposure']:
        exposure = (camera.exposure._exposure if fix_exposure is None else fix_exposure)
        render_pkg['render_color'] = exposure[0] * render_pkg['render_color'] + exposure[1]
    if config['render']['use_weight_norm']:
        render_pkg['render_depth'] = render_pkg['render_depth'] / (render_pkg['render_alpha'] + config['render']['eps'])
        outliner_mask = torch.logical_or(render_pkg['render_depth'] > config['render']['depth_far'], render_pkg['render_depth'] < config['render']['depth_near']) 
        render_pkg['render_depth'][outliner_mask] = 0
    return render_pkg

def Renderer_BA(config: dict, gaussians:Gaussians, camera:Frame = None, fix_w2c = None, fix_exposure = None):
    method = config['render']['method']
    h = config['cameras']['height']
    w = config['cameras']['width']
    intrinsics = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
    
    view_offset = torch.eye(4, dtype=torch.float32, device='cuda')
    if method == '2dgs' and not config['render']['use_sa']:
        cam = setup_camera[method](w, h, intrinsics, view_offset, use_sa=False)
    else: 
        cam = setup_camera[method](w, h, intrinsics, view_offset)
    params = gaussians.get_render_params
    params.pop('features', None)
    w2c = camera.get_w2c if fix_w2c is None else fix_w2c
    params['means3D'] = (w2c[:3, :3] @ params['means3D'].T  + w2c[:3, 3:]).T
    params['rotations'] = quaternion_multiply(matrix_to_quaternion(w2c[None, :3, :3]), params['rotations']).detach()
    points2D = torch.zeros_like(gaussians.get_xyz, requires_grad=True)
    points2D.retain_grad()
    params['means2D'] = points2D
    render_pkg = render[method](cam, **params)
    
    if config['render']['enable_exposure']:
        exposure = (camera.exposure._exposure if fix_exposure is None else fix_exposure)
        render_pkg['render_color'] = exposure[0] * render_pkg['render_color'] + exposure[1]
    if config['render']['use_weight_norm']:
        render_pkg['render_depth'] = render_pkg['render_depth'] / (render_pkg['render_alpha'] + config['render']['eps'])
        outliner_mask = torch.logical_or(render_pkg['render_depth'] > config['render']['depth_far'], render_pkg['render_depth'] < config['render']['depth_near']) 
        render_pkg['render_depth'][outliner_mask] = 0
    return render_pkg

def Renderer_view(config: dict, gaussians:Gaussians, w2c, scale = 1.0, sample=1.0, render_feat=False):
    method = config['render']['method']
    h = config['cameras']['height']
    w = config['cameras']['width']
    intrinsics = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
    view_offset = torch.eye(4, dtype=torch.float32, device='cuda')
    if method == '2dgs' and not config['render']['use_sa']:
        cam = setup_camera[method](w, h, intrinsics, view_offset @ w2c, use_sa=False)
    else:
        cam = setup_camera[method](w, h, intrinsics, view_offset @ w2c)

    params = gaussians.get_render_params
    if not render_feat or params.get('features', None) is None:
        params.pop('features', None)
    else:
        params.update(feature_render_args(config))
    points2D = torch.zeros_like(gaussians.get_xyz, requires_grad=True)
    points2D.retain_grad()
    params['means2D'] = points2D
    render_pkg = render[method](cam, **params)
    if config['render']['use_weight_norm']:
        render_pkg['render_depth'] = render_pkg['render_depth'] / (render_pkg['render_alpha'] + config['render']['eps'])
        outliner_mask = torch.logical_or(render_pkg['render_depth'] > config['render']['depth_far'], render_pkg['render_depth'] < config['render']['depth_near']) 
        render_pkg['render_depth'][outliner_mask] = 0
    return render_pkg
