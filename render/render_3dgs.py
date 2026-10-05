import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings as RenderCamera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from .feature_render import render_feature_map, feature_camera

def setup_camera(w, h, k, w2c, near=0.01, far=100) -> RenderCamera:
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
        prefiltered=False,
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

    # print("debug", scales.shape)
    if (torch.isnan(means3D).any() or torch.isnan(rotations).any()) or (torch.isnan(opacities).any() or torch.isnan(colors_precomp).any()):
        import pdb 
        pdb.set_trace()
    color_map, radius, _ = Renderer(rendercamera)(
        means3D,
        means2D,
        opacities = opacities,
        shs = shs,
        colors_precomp = colors_precomp,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )
    
    # splatam
    pts_3D = means3D 
    w2c = rendercamera.viewmatrix.squeeze().T
    
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    sil_map, _, _ = Renderer(rendercamera)(
        means3D,
        means2D,
        opacities = opacities,
        shs = shs,
        colors_precomp = depth_silhouette,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )

    render_pkg = {
        "render_color": color_map,
        "radius": radius,
        "means2D": means2D,
        "render_depth": sil_map[0:1],
        "render_alpha": sil_map[1:2],
        "render_normal": torch.zeros_like(color_map),
        "render_middepth": torch.zeros_like(sil_map[0:1]),
        "render_dist": torch.zeros_like(sil_map[0:1]),
    }
    if features is not None:
        render_pkg["render_feat"] = render_feature_map(Renderer(feature_camera(rendercamera, feat_scale)),
                                                       means3D, means2D, opacities, scales, rotations, cov3D_precomp,
                                                       features, visible=(radius > 0), use_checkpoint=feat_checkpoint,
                                                       detach_geometry=feat_detach_geometry)
    return render_pkg
    