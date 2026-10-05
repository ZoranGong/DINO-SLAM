from utils.common_utils import build_rotation, build_quaternion
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH, SH2RGB

import torch
import numpy as np
import torch.nn as nn
from simple_knn._C import distCUDA2
from collections import OrderedDict
import os 
from plyfile import PlyData, PlyElement
from utils.common_utils import viewmatrix


class Gaussians:
    
    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
    @staticmethod
    def inverse_opacity_activation(x):
        return torch.log(x/(1-x))
    
    def __init__(self, config: dict) -> None:
        
        self.config = config
        
        self.method = config['render']['method']
        if self.method not in ['2dgs', '3dgs', 'radegs']:
            raise ValueError(f'''Unknown rendering method {self.method} !!! ''')
        
        self.gaussian_distribution = config['gaussians']['gaussian_distribution']
        if self.gaussian_distribution not in ['isotropic', 'anisotropic']:
            raise ValueError(f'''Unknown gauasian_distribution {self.gaussian_distribution} !!! ''')
        
        self.use_sh = False # not support
        if self.use_sh:
            self.max_sh_degree = config['gaussians']['sh']['sh_degree']
        self.setup_functions()

        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        
        self._features_dc = None
        self._features_rest = None
        self._rgb = None
        self.optimizer = None

        # GeoDINO feature attribute (slam/SGE.py)
        sge_cfg = config.get('sge', {})
        self.use_feat = bool(sge_cfg.get('enable', False))
        self.feat_dim = int(sge_cfg.get('feat_dim', 16)) if self.use_feat else 0
        self._feat = None

    def _init_feat(self, num_pts, feats=None):
        if feats is not None:
            feats = feats.float().cuda()
            assert feats.shape == (num_pts, self.feat_dim)
            return feats.contiguous()
        return torch.zeros((num_pts, self.feat_dim), dtype=torch.float32, device='cuda')

    def add_densification_stats(self, render_pkg):
        update_filter = render_pkg["radius"] > 0
        viewspace_point_tensor = render_pkg["means2D"]
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1)
        self.denom[update_filter] += 1
    
    def create_from_pcd(self, pcd : BasicPointCloud, initial_scale = None, spatial_lr_scale: float=0.0):
        # self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = pcd.points.float().cuda()
        
        if self.use_sh:
            fused_rgb = RGB2SH(pcd.colors.float().cuda())
            features = torch.zeros((fused_rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_rgb
            features[:, 3:, 1:] = 0.0
        else:
            precomp_rgb = pcd.colors.float().cuda()
            
        if initial_scale is None:
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.detach().clone().float().cuda()), 0.0000001)
            initial_scale = torch.sqrt(dist2)
        
        if self.gaussian_distribution == 'isotropic':
            scales = torch.log(initial_scale)[...,None].repeat(1, 1)
        else:
            if self.method == '2dgs':
                scales = torch.log(initial_scale)[...,None].repeat(1, 2)
            else:
                scales = torch.log(initial_scale)[...,None].repeat(1, 3)
        
        if self.method == '2dgs' and (pcd.normals is not None):
            view_dir = pcd.normals.float().cuda()
            up       = torch.stack([view_dir[:, 1] * view_dir[:, 2],
                                    view_dir[:, 0] * view_dir[:, 2],
                                    -2 * view_dir[:, 0] * view_dir[:, 1]], dim=-1)
            rotation_matrixs = viewmatrix(view_dir, up)
            rots = build_quaternion(rotation_matrixs)
            rots = torch.nan_to_num(rots, 0, 0)
            mask = rots.norm(dim=-1) < 1e-3
            t = torch.zeros((mask.sum(), 4), dtype=torch.float32, device='cuda') 
            t[:, 0] = 1
            rots[mask] = t
        else: 
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            
        opacities = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")# self.inverse_opacity_activation(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        if self.use_sh:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        else:
            self._rgb = nn.Parameter(precomp_rgb.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if self.use_feat:
            self._feat = nn.Parameter(self._init_feat(fused_point_cloud.shape[0], pcd.features).requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.training_setup()

    def training_setup(self, ):
        training_args = self.config['gaussians']['training_args']

        l = [
            {'params': [self._xyz], 'lr': training_args['xyz_lr'], "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args['rotation_lr'], "name": "rotation"},
        ]

        if self.use_sh:
            l.append({'params': [self._features_dc], 'lr': training_args['feature_lr'], "name": "f_dc"})
            l.append({'params': [self._features_rest], 'lr': training_args['feature_lr'] / 20.0, "name": "f_rest"})
        else:
            l.append({'params': [self._rgb], 'lr': training_args['rgb_lr'], "name": "rgb"})
        if self.use_feat:
            l.append({'params': [self._feat], 'lr': training_args.get('feat_lr', training_args['rgb_lr']), "name": "feat"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
    
    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] in ["camsh", "camrots", "camtrans"]: continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def add_gaussians_from_pcd(self, pcd: BasicPointCloud, inital_scales=None):
        num_pts = pcd.points.shape[0]
        if isinstance(pcd.points, torch.Tensor):
            new_xyz = pcd.points
            new_col = pcd.colors
            new_norm = pcd.normals
        else:
            new_xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
            new_col = torch.tensor(np.asarray(pcd.colors)).float().cuda()
            new_norm = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        
        num_pts = pcd.points.shape[0]        
        if self.method == '2dgs' and (pcd.normals is not None):
            view_dir = new_norm
            up       = torch.stack([view_dir[:, 1] * view_dir[:, 2],
                                    view_dir[:, 0] * view_dir[:, 2],
                                    -2 * view_dir[:, 0] * view_dir[:, 1]], dim=-1)
            rotation_matrixs = viewmatrix(view_dir, up)

            new_rots = build_quaternion(rotation_matrixs)
            new_rots = torch.nan_to_num(new_rots, 0, 0)
            mask = new_rots.norm(dim=-1) < 1e-3
            t = torch.zeros((mask.sum(), 4), dtype=torch.float32, device='cuda') 
            t[:, 0] = 1
            new_rots[mask] = t
        else: 
            new_rots = torch.zeros((num_pts, 4), device="cuda", dtype=torch.float32)
            new_rots[:, 0] = 1
            
        new_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device='cuda')
        
        if inital_scales is None:
            dist2 = torch.clamp_min(distCUDA2(new_xyz.detach().clone().float().cuda()), 0.0000001)
            inital_scales = torch.sqrt(dist2)
        if self.gaussian_distribution == 'isotropic':
            new_log_scales = torch.tile(torch.log(inital_scales)[..., None], (1, 1))
        else:
            if self.method == '2dgs':
                new_log_scales = torch.tile(torch.log(inital_scales)[..., None], (1, 2))
            else:
                new_log_scales = torch.tile(torch.log(inital_scales)[..., None], (1, 3))
        
        new_params = {
            'xyz': new_xyz,
            'opacity': new_opacities,
            'scaling': new_log_scales,
            'rotation': new_rots,
        }
        
        if self.use_sh:
            fused_rgb = RGB2SH(new_col)
            features = torch.zeros((fused_rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_rgb
            features[:, 3:, 1:] = 0.0
            
            new_features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
            new_features_rest = features[:,:,1:].transpose(1, 2).contiguous()
            new_params.update({'f_dc': new_features_dc, 'f_rest': new_features_rest})
        else:
            new_rgb = new_col
            new_params.update({'rgb': new_rgb})
        if self.use_feat:
            new_params.update({'feat': self._init_feat(num_pts, pcd.features)})

        for k, v in new_params.items():
        # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                new_params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            else:
                new_params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
        
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)
        
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        if self.use_sh:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        else:
            self._rgb = optimizable_tensors["rgb"]
        if self.use_feat:
            self._feat = optimizable_tensors["feat"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def zero_grad(self, mask=None, t=None):
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if t is not None:
                group["params"][0][:t].grad = None
            else:
                group["params"][0][mask].grad = None
        
    def remove_gaussians_from_mask(self, mask):
        valid_points_mask = ~mask 
        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        if self.use_sh:        
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        else:
            self._rgb = optimizable_tensors["rgb"]
        if self.use_feat:
            self._feat = optimizable_tensors["feat"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_feat(self):
        return self._feat if self.use_feat else None

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1) if self.use_sh else None
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_rgb(self):
        return self._rgb if not self.use_sh else None
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def size(self):
        return self._xyz.shape[0]
    
    @property
    def get_render_params(self,):
        
        if self.gaussian_distribution == 'isotropic':
            if self.method == '2dgs':
                scales = torch.tile(self.get_scaling, (1, 2))
            else:
                scales = torch.tile(self.get_scaling, (1, 3))
        else:
            scales = self.get_scaling
        # print(scales.shape, self.get_scaling.shape)
        return OrderedDict(
            means3D = self.get_xyz,
            means2D = None,
            opacities = self.get_opacity,
            shs = self.get_features,
            colors_precomp = self.get_rgb,
            scales = scales,
            rotations = self.get_rotation,
            cov3D_precomp = None,
            features = self.get_feat,
        )

    def extract_params(self):
        params = {
            "xyz": self._xyz.detach(),
            "opacity": self._opacity.detach(),
            "scaling": self._scaling.detach(),
            "rotation": self._rotation.detach(),
        }
        if self.use_sh:
            params.update({"f_dc": self._features_dc.detach(), "f_rest": self._features_rest.detach()})
        else:
            params.update({"rgb": self._rgb.detach()})
        if self.use_feat:
            params.update({"feat": self._feat.detach()})
        return params

    def create_params(self, params):
        self._xyz = nn.Parameter(params["xyz"].requires_grad_(True))
        if self.use_sh:
            self._features_dc = nn.Parameter(params["f_dc"].requires_grad_(True))
            self._features_rest = nn.Parameter(params["f_rest"].requires_grad_(True))
        else:
            self._rgb = nn.Parameter(params["rgb"].requires_grad_(True))
        self._scaling = nn.Parameter(params["scaling"].requires_grad_(True))
        self._rotation = nn.Parameter(params["rotation"].requires_grad_(True))
        self._opacity = nn.Parameter(params["opacity"].requires_grad_(True))
        if self.use_feat:
            feat = params["feat"] if "feat" in params else self._init_feat(self._xyz.shape[0])
            self._feat = nn.Parameter(feat.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.training_setup()
        
    def add_params(self, new_params):

        num_pts = new_params["xyz"].shape[0]
        if self.use_feat and "feat" not in new_params:
            new_params["feat"] = self._init_feat(num_pts)
        optimizable_tensors = self.cat_tensors_to_optimizer(new_params)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.use_sh:
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
        else:
            self._rgb = optimizable_tensors["rgb"]
        if self.use_feat:
            self._feat = optimizable_tensors["feat"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        return self._xyz.shape[0]
    
    def reset_opacity(self, div=0):
        
        opacities_new = self.get_opacity.detach()
        opacities_new[div:, :] = torch.min(opacities_new[div:, :], 
                                           self.inverse_opacity_activation(0.01 * torch.ones_like(opacities_new[div:, :])))
         
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    @torch.no_grad()
    def transform_gaussians(self, transfer):
        new_xyz = (transfer[:3, :3] @ self._xyz.T  + transfer[:3, 3:]).T
        new_rot = build_quaternion(torch.matmul(transfer[None, :3, :3], build_rotation(self._rotation)))
        
        optimizable_tensors = self.replace_tensor_to_optimizer(new_xyz, "xyz")
        self._xyz = optimizable_tensors['xyz']
        
        optimizable_tensors = self.replace_tensor_to_optimizer(new_rot, "rotation")
        self._rotation = optimizable_tensors["rotation"]
        
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
            
        if self.use_sh:
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        else:
            l = l + list('rgb')
        if self.use_feat:
            for i in range(self.feat_dim):
                l.append('feat_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        normals = torch.zeros_like(self._xyz)
        params_list = [self._xyz, normals, self._opacity, self._scaling, self._rotation]
        params_list = params_list + ([self._features_dc, self._features_rest] if self.use_sh else [self._rgb])
        if self.use_feat:
            params_list = params_list + [self._feat]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(self._xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate([params.detach().cpu().numpy() for params in params_list], axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if self.use_sh:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self.active_sh_degree = self.max_sh_degree
        else:
            rgb = np.stack((np.asarray(plydata.elements[0]["r"]),
                        np.asarray(plydata.elements[0]["g"]),
                        np.asarray(plydata.elements[0]["b"])),  axis=1)
            self._rgb = nn.Parameter(torch.tensor(rgb, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_feat:
            feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("feat_")]
            feat_names = sorted(feat_names, key = lambda x: int(x.split('_')[-1]))
            feats = np.zeros((xyz.shape[0], self.feat_dim))
            for idx, attr_name in enumerate(feat_names[:self.feat_dim]):
                feats[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._feat = nn.Parameter(torch.tensor(feats, dtype=torch.float, device="cuda").requires_grad_(True))
        return self

    def densification_postfix(self, new_xyz, new_rgb, new_opacities, new_scaling, new_rotation, new_feat=None):
        d = {"xyz": new_xyz,
        "rgb": new_rgb,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.use_feat:
            d["feat"] = new_feat if new_feat is not None else self._init_feat(new_xyz.shape[0])

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._rgb = optimizable_tensors["rgb"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_feat:
            self._feat = optimizable_tensors["feat"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(grads >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.config['densify']['percent_dense']*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        if self.use_sh:
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
        else:
            new_rgb = self._rgb[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_feat = self._feat[selected_pts_mask] if self.use_feat else None

        self.densification_postfix(new_xyz, new_rgb, new_opacities, new_scaling, new_rotation, new_feat)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.config['densify']['percent_dense']*scene_extent)
        if selected_pts_mask.sum() == 0: return
        scales = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([scales, torch.zeros((scales.shape[0], 1), dtype=torch.float32, device='cuda')], dim=-1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_rgb = self._rgb[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_feat = self._feat[selected_pts_mask].repeat(N, 1) if self.use_feat else None

        self.densification_postfix(new_xyz, new_rgb, new_opacity, new_scaling, new_rotation, new_feat)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.remove_gaussians_from_mask(prune_filter)

    def densify_and_prune(self, ):
        max_grad = self.config['densify']['densify_grad_threshold']
        min_opacity = self.config['densify']['opacity_cuil']
        extent = self.config['densify']['extent']
        max_screen_size = self.config['densify']['scale_max']
        min_scale = self.config['densify']['scale_cuil']
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask = prune_mask | (self.get_scaling.mean(dim=-1) < min_scale)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.remove_gaussians_from_mask(prune_mask)

        torch.cuda.empty_cache()