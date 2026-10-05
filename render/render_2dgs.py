import torch

from gaus_2dgs_rasterization import GaussianRasterizationSettings as RenderCamera
from gaus_2dgs_rasterization import GaussianRasterizer as Renderer
from .feature_render import render_feature_map, feature_camera

def setup_camera(w, h, k, w2c, near=0.01, far=100, use_sa=True) -> RenderCamera:
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = w2c.cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = RenderCamera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        use_sa=use_sa,
        prefiltered=False,
        debug=False,
    )
    return cam

def render(
    rendercamera: RenderCamera,
    
    means3D,
    means2D,
    opacities,
    shs = None,    
    colors_precomp = None,
    scales = None,
    rotations = None,
    cov3D_precomp = None,
    features = None,
    feat_checkpoint = False,
    feat_scale = 1.0,
    feat_detach_geometry = True,
):
    color_map, radius, allmap = Renderer(rendercamera)(
        means3D,
        means2D,
        opacities = opacities,
        shs = shs,
        colors_precomp = colors_precomp,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    render_pkg = {
        "render_color": color_map,
        "radius": radius,
        "means2D": means2D,
        "render_depth": allmap[0:1],
        "render_alpha": allmap[1:2],
        "render_normal": allmap[2:5],
        "render_middepth": allmap[5: 6],
        "render_dist": allmap[6:7],
    }
    if features is not None:
        render_pkg["render_feat"] = render_feature_map(Renderer(feature_camera(rendercamera, feat_scale)),
                                                       means3D, means2D, opacities, scales, rotations, cov3D_precomp,
                                                       features, visible=(radius > 0), use_checkpoint=feat_checkpoint,
                                                       detach_geometry=feat_detach_geometry)
    return render_pkg
    