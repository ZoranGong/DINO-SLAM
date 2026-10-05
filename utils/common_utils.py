import os

import numpy as np
import random
import pytorch3d.transforms
import torch

from utils.graphics_utils import BasicPointCloud
import pytorch3d


def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")

from scipy.spatial.transform import Rotation as R
def build_quaternion(rotation_matrix):
    return pytorch3d.transforms.matrix_to_quaternion(rotation_matrix)

def quaternion_mut(q1: torch.Tensor, q2: torch.Tensor):
    q1 = q1 / (q1.norm(dim=1)[:, None] + 1e-5)
    q2 = q2 / (q2.norm(dim=1)[:, None] + 1e-5)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([w, x, y, z], dim=1)

def build_rotation(r: torch.Tensor):
    R = pytorch3d.transforms.quaternion_to_matrix(r)
    # q = r / r.norm(dim=1)[:, None]

    # R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # r = q[:, 0]
    # x = q[:, 1]
    # y = q[:, 2]
    # z = q[:, 3]

    # R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    # R[:, 0, 1] = 2 * (x*y - r*z)
    # R[:, 0, 2] = 2 * (x*z + r*y)
    # R[:, 1, 0] = 2 * (x*y + r*z)
    # R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    # R[:, 1, 2] = 2 * (y*z - r*x)
    # R[:, 2, 0] = 2 * (x*z - r*y)
    # R[:, 2, 1] = 2 * (y*z + r*x)
    # R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def viewmatrix(lookdir: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Construct lookat view matrix. """
    vec2 = lookdir / lookdir.norm(dim=-1)[:, None]
    vec0 = torch.cross(up, vec2, dim=-1)
    vec0 = vec0 / vec0.norm(dim=-1)[:, None]
    vec1 = torch.cross(vec2, vec0, dim=-1)
    vec1 = vec1 / vec1.norm(dim=-1)[:, None]
    m = torch.stack([vec0, vec1, vec2], dim=-1)
    return m

def get_normalmask_from_depth(H, W, depth: torch.Tensor, near=0.01, far=15.0):
    '''
    input:
        H: image height
        W: image width
        depth: depth map [H, W]
    return:
        normalmask: [N]
    '''
    depth_mask = (depth > near) & (depth < far)
    depth_mask = depth_mask.reshape(H, W)
    normal_mask = depth_mask
    normal_mask[1:, :] = normal_mask[1:, :] & depth_mask[:-1, :]
    normal_mask[:, 1:] = normal_mask[:, 1:] & depth_mask[:, :-1]
    normal_mask[:-1, :] = normal_mask[:-1, :] & depth_mask[1:, :]
    normal_mask[:, :-1] = normal_mask[:, :-1] & depth_mask[:, 1:]
    return normal_mask.reshape(-1)

def get_projected_pts(H, W, intrinsics, pts):
    
    CX, CY, FX, FY = intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1]
    depth_z = pts[:, 2:3]
    uv = pts[:, :2] / (depth_z + 1e-5)

    uv[:, 0] = uv[:, 0] * FX + CX
    uv[:, 1] = uv[:, 1] * FY + CY
    
    mask_u = (uv[:, 0] > 0) & (uv[:, 0] < W) 
    mask_v = (uv[:, 1] > 0) & (uv[:, 1] < H)
    
    uv[:, 0] = 2 * uv[:, 0] / (W - 1 + 1e-5) - 1.0
    uv[:, 1] = 2 * uv[:, 1] / (H - 1 + 1e-5) - 1.0
    return uv, depth_z, (mask_u & mask_v & (depth_z[:, 0] > 1e-5))

# depth [H, W]
def get_pts_from_depth(H, W, intrinsics, depth: torch.Tensor, down_sample=1):
    depth = depth.reshape(H, W)
    
    CX, CY, FX, FY = intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1]
    
    if down_sample > 1:
        CX = CX // down_sample
        CY = CY / down_sample
        FX = FX / down_sample
        FY = FY / down_sample
        H = H // down_sample
        W = W // down_sample
        depth = torch.nn.functional.interpolate(depth[None, None,...], size=(H, W), mode='bilinear', align_corners=True)[0, 0]
        
    x_grid, y_grid = torch.meshgrid(torch.arange(W, dtype=torch.float32, device='cuda'), 
                                     torch.arange(H, dtype=torch.float32, device='cuda'), indexing='xy')
    
    xx = (x_grid - CX) / FX 
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)
    pts = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    return pts


def transform_pts_by_homo(pts, homo):
    '''
    input:
        pts: [N, 3]
        homo: [4, 4]
    return:
        transformed_pts: [N, 3]
    '''
    
    pix_ones = torch.ones(pts.shape[0], 1, dtype=torch.float32, device='cuda')
    pts4 = torch.cat((pts, pix_ones), dim=-1)
    transformed_pts = (homo @ pts4.T).T[:, :3]
    return transformed_pts 

def transform_pts_by_rt(pts, R, T):
    '''
    input:
        pts: [N, 3]
        R: [3, 3]
        T: [3]S
    return:
        transformed_pts: [N, 3]
    '''
    transformed_pts = (R @ pts.T + T[:, None]).T
    return transformed_pts 

def get_normal_from_pts(H, W, pts):
    '''
    input:
        H: image height
        W: image width
        pts: pointcloud from image
    return:
        normal: [N, 3]
    '''
    pts = pts.reshape(H, W, 3)
    normal = torch.rand_like(pts)
    dx = torch.cat([pts[2:, 1:-1] - pts[:-2, 1:-1]])
    dy = torch.cat([pts[1:-1, 2:] - pts[1:-1, :-2]], dim=1) 
    normal_map = torch.cross(dx, dy, dim=-1)
    normal[1:-1, 1:-1, :] = normal_map
    normal = torch.nn.functional.normalize(normal, dim=-1)
    return normal.reshape(-1, 3)

def get_mean3_sq_dist(H, W, intrinsics, depth):
    '''
    mean3_sq_dist used for initializing gaussian scales
    input:
        H: image height
        W: image width
        pts: pointcloud from image
    return:
        normal: [N, 3]
    '''
    depth_z = depth.reshape(-1)
    CX, CY, FX, FY = intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1]
    
    scales_guassian = depth_z / ((FX + FY) / 2)
    mean3_sq_dist = (scales_guassian ** 2)
    return torch.sqrt(mean3_sq_dist)
    
def get_pointcloud(color, depth, intrinsics, c2w = None, w2c = None, color_mask = None, compute_mean_sq_dist = False, sample_num=None, feat=None):
    # feat: optional per-pixel feature map [H, W, C], sampled with the same indices as pts / col / norm
    if w2c is not None:
        c2w = torch.linalg.inv(w2c)

    W, H = color.shape[1], color.shape[0]

    pts = get_pts_from_depth(H, W, intrinsics, depth)
    mask = get_normalmask_from_depth(H, W, depth)

    if color_mask is not None:
        mask = mask & color_mask.reshape(-1)

    if c2w is not None:
        pts = transform_pts_by_homo(pts, c2w)

    normal = get_normal_from_pts(H, W, pts)
    pts = pts.reshape(-1, 3)[mask]
    col = color.reshape(-1, 3)[mask]
    norm = normal.reshape(-1, 3)[mask]
    ft = None
    if feat is not None:
        assert feat.shape[0] == H and feat.shape[1] == W
        ft = feat.reshape(H * W, -1)[mask]

    sample_idxs = None
    if sample_num is not None and sample_num < len(pts):
        sample_idxs = random.sample(range(len(pts)), sample_num)
        pts = pts[sample_idxs]
        col = col[sample_idxs]
        norm = norm[sample_idxs]
        if ft is not None:
            ft = ft[sample_idxs]

    pcd = BasicPointCloud(points=pts, colors=col, normals=norm, features=ft)
    if compute_mean_sq_dist:
        inital_scale = get_mean3_sq_dist(H, W, intrinsics, depth)[mask]
        if sample_idxs is not None:
            inital_scale = inital_scale[sample_idxs]
        return pcd, inital_scale
    return pcd
    
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper