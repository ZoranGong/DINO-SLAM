import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.common_utils import build_rotation, build_quaternion
import numpy as np
import random
from utils.descriptor import MyDesc
import copy 

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1, max_steps=1000000
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
        log_lerp = (1 - t) * lr_init + t * lr_final # np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class Transform:
    def __init__(self, adam_betas: tuple = (0.9, 0.99)):
        self.optimizer = None 
        self.adam_betas = adam_betas
        self.freeze = False
    
    def set_freeze(self, ):
        self.freeze = True 
        self.update_learning_rate(step=False)
    
    def reset_freeze(self, ):
        self.freeze = False 
        self.update_learning_rate(step=False)
        
    def init_optimizer(self, lr_dict, initial_transformation=None):
        
        if initial_transformation is not None:
            rot = build_quaternion(initial_transformation[:3, :3][None, ...])[0]
            trans = initial_transformation[:3, 3]
        else: 
            rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device='cuda')
            trans = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
            
        self._cam_rot = nn.Parameter(rot.requires_grad_(True))
        self._cam_trans = nn.Parameter(trans.requires_grad_(True))
        self.optimizer = torch.optim.Adam(
            [{'params': [self._cam_rot], 'lr': lr_dict['cam_rot_lr_init'], "name": f"cam_rots"},
             {'params': [self._cam_trans], 'lr': lr_dict['cam_trans_lr_init'] , "name": f"cam_trans"}], 
            lr=0.0, eps=1e-8, betas=self.adam_betas)
        self.cam_rot_scheduler_args = get_expon_lr_func(lr_init=lr_dict['cam_rot_lr_init'],
                                                        lr_final=lr_dict['cam_rot_lr_final'],
                                                        max_steps=lr_dict['cam_rot_lr_max_step'])
        self.cam_trans_scheduler_args = get_expon_lr_func(lr_init=lr_dict['cam_trans_lr_init'],
                                                          lr_final=lr_dict['cam_trans_lr_final'],
                                                          max_steps=lr_dict['cam_trans_lr_max_step'])
        self.iteration_times = 0
        self.update_learning_rate(step=False)
        
    @property
    def get_transform_matrix(self):
            
        cam_rot = F.normalize(self._cam_rot[None, ...])[0]
        cam_tran = self._cam_trans
        
        w2c = torch.eye(4, dtype=torch.float32, device='cuda')
        w2c[:3, :3] = build_rotation(cam_rot[None, ...])[0]
        w2c[:3, 3] = cam_tran
        return w2c
    
    def update_learning_rate(self, step=True):
        if step: self.iteration_times = self.iteration_times + 1
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "cam_rots":
                lr = self.cam_rot_scheduler_args(self.iteration_times) if not self.freeze else 0.0
                param_group['lr'] = lr
            if param_group["name"] == "cam_trans":
                lr = self.cam_trans_scheduler_args(self.iteration_times) if not self.freeze else 0.0
                param_group['lr'] = lr   

class Exposure:
    def __init__(self, adam_betas: tuple = (0.9, 0.99)):
        self.optimizer = None 
        self.adam_betas = adam_betas
        self.freeze = False
        
    def set_freeze(self, ):
        self.freeze = True 
        self.update_learning_rate()
    
    def reset_freeze(self, ):
        self.freeze = False 
        self.update_learning_rate()

    def init_optimizer(self, lr_dict):
        
        exposure = torch.tensor([[1], [0]], device='cuda', dtype=torch.float32) 
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        self.lr_dict = lr_dict        
        self.optimizer = torch.optim.Adam(
            [{'params': [self._exposure], 'lr': lr_dict['exposure_lr_init'], "name": "exposure"}], 
            lr=0.0, eps=1e-8, betas=self.adam_betas)

        self.exposure_scheduler_args = get_expon_lr_func(lr_init=lr_dict['exposure_lr_init'],
                                                         lr_final=lr_dict['exposure_lr_final'],
                                                         max_steps=lr_dict['exposure_lr_max_step'])
        self.iteration_times = 0
        self.update_learning_rate(step=False)
    
    def update_learning_rate(self, step=True):
        if step: self.iteration_times = self.iteration_times + 1
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "exposure":
                lr = self.exposure_scheduler_args(self.iteration_times) if not self.freeze else 0.0
                param_group['lr'] = lr
    
class Frame:
    def __init__(self, config: dict, time_idx: int, gt_color: torch.Tensor, gt_depth: torch.Tensor, gt_pose: torch.Tensor, 
                 kfid, frame_type=0) -> None:
        
        self.h = config['cameras']['height']
        self.w = config['cameras']['width']
        self.intrinsics = torch.tensor(config['cameras']['intrinsics'], dtype=torch.float32, device='cuda')
        self.enable_exposure = config['render']['enable_exposure']
        
        self.adam_betas = config['cameras']['adam_betas']
        self.frame_type = frame_type
        self.time_idx = time_idx
        self.gt_color = gt_color
        self.gt_depth = gt_depth
        self.gt_pose = gt_pose 
        self.kfid = kfid
        self.gt_w2c = torch.linalg.inv(gt_pose)
        
        self.transform = None 
        self.exposure = None        
        self.est_w2c = torch.eye(4, dtype=torch.float32, device='cuda')
        self.mapping_times = 0
        if self.enable_exposure:
            self.est_exposure = torch.tensor([[1], [0]], dtype=torch.float32, device='cuda')
        
    def start_optimizer(self, initial_transformation: torch.Tensor, lr_dict: dict):
        
        self.transform = Transform(adam_betas=self.adam_betas)
        self.transform.init_optimizer(lr_dict, initial_transformation)
        
        if self.enable_exposure:
            self.exposure = Exposure()
            self.exposure.init_optimizer(lr_dict)
            
    
    def finish_optimizer(self, save=False):

        self.est_w2c = self.get_w2c.clone().detach()
        self.transform = None 
        
        if self.enable_exposure:
            self.est_exposure = self.exposure._exposure.clone().detach()
            self.exposure = None
        
        if not save:
            self.gt_color = None 
            self.gt_depth = None
            self.gt_normal = None 
            
    def to_device(self, device):
        # move the stored RGB-D between CPU and GPU
        for name in ('gt_color', 'gt_depth'):
            t = getattr(self, name, None)
            if t is not None and t.device.type != device:
                setattr(self, name, t.to(device, non_blocking=True))
        return self

    @property
    def get_w2c(self):
        if self.transform is not None:
            return self.transform.get_transform_matrix
        return self.est_w2c
    
    @property
    def get_exposure(self):
        if self.exposure is not None:
            return self.exposure._exposure
        return self.est_exposure
        
        
class LocalMap:
    def __init__(self, config: dict, lmid, frames: list, local_map_params: dict, tracking_ok=True) -> None:
        self.config = copy.deepcopy(config)
        self.lmid = lmid 
        self.frames = frames
        self.enable_exposure = config['render']['enable_exposure']
        self.tracking_ok = tracking_ok
        # saving frames
        num_saved = self.config['backend']['num_frame_saved']
        frame_pri = [random.randint(0, 100) for frame in frames[:-1]]
        frame_pri[0] = frame_pri[0] + 400
        frame_pri[-1] = frame_pri[-1] + 400
        for i in range(len(frame_pri)):
            frame_pri[i] = (frames[i].frame_type < 2) * 200 + frame_pri[i]
        frame_id = list(range(len(frame_pri)))
        frame_id = sorted(frame_id, reverse=True, key=lambda x: frame_pri[x])
        self.saved_idxs = frame_id[: min(num_saved, len(frame_id))]
        
        self.ref2f0 = frames[0].get_w2c.detach()
        f02ref = torch.linalg.inv(self.ref2f0)
        for idx, frame in enumerate(frames):
            frame.finish_optimizer(save=(idx in self.saved_idxs))
            frame.est_w2c = frame.est_w2c @ f02ref
        
        self.transform = None 
        self.exposure = None 
        self.local_map_params = local_map_params
        
        # calc localmap desc
        num_frames = len(frames) 
        rep_imgs = torch.stack([self.frames[0].gt_color.permute(2, 0, 1),
                                self.frames[num_frames - 2].gt_color.permute(2, 0, 1)])
        self.map_desc = MyDesc()(rep_imgs)
        self.mapping_times = 0
        
    def start_optimizer(self, initial_transformation: torch.Tensor, lr_dict: dict):
        
        self.transform = Transform(adam_betas=(0.7, 0.99))
        self.transform.init_optimizer(lr_dict, initial_transformation)
        
        if self.enable_exposure:
            self.exposure = Exposure()
            self.exposure.init_optimizer(lr_dict)
    
    def offload(self, device='cpu'):
        for idx in self.saved_idxs:
            self.frames[idx].to_device(device)
        return self

    def get_frame_w2c(self, f_idx):
        assert (self.transform is not None) and f_idx < len(self.frames)
        return self.frames[f_idx].get_w2c @ self.transform.get_transform_matrix

    def get_frame_exposure(self, f_idx):
        if self.exposure is None:
            return torch.tensor([[1], [0]], dtype=torch.float32, device='cuda')
        
        frame: Frame = self.frames[f_idx]
        exposure_A = self.exposure._exposure[0:1, :1] * frame.est_exposure[0:1, :1]
        exposure_B = self.exposure._exposure[0:1, :1] * frame.est_exposure[1:2, :1] + self.exposure._exposure[1:2, :1]
        return torch.cat([exposure_A, exposure_B], dim=0)
    
    @property
    def get_w2c(self, ):
        assert self.transform is not None
        return self.transform.get_transform_matrix
        
class Localmaps(list):
    
    def __init__(self, config):
        self.config = config
        self.map_descs = None
        super(Localmaps, self).__init__()

    def frames_size(self, lm_idx):
        
        size = len(self[lm_idx].frames)
        return size
    
    def add_localmap(self, lm: LocalMap):
        self.append(lm)
        
        if self.map_descs == None:
            self.map_descs = lm.map_desc[None]
        else:
            self.map_descs = torch.cat([self.map_descs, lm.map_desc[None]])
    
    def query_covisable(self, lm_idx, num_kf=10):
        
        query_desc = self.map_descs[lm_idx]
        
        i, k, d = self.map_descs.shape
        sims = torch.einsum("id,jd->ij", self.map_descs.view(i * k, d), query_desc).view(i, -1)
        max_sims, _ = torch.max(sims, dim=1)
        
        score, max_sim_lmidxs = max_sims.topk(min(num_kf, i))
        return max_sim_lmidxs.tolist()
    
    def get_frame_w2c(self, lm_idx, f_idx):
        return self[lm_idx].get_frame_w2c(f_idx)
    
    def get_w2cs(self):
        w2cs = []
        for lm in self:
            for f in lm.frames[:-1]:
                if f.time_idx == len(w2cs) and lm.transform is not None:
                    w2cs.append((f.get_w2c @ lm.get_w2c).detach())
        w2cs.append((self[-1].frames[-1].get_w2c @ self[-1].get_w2c).detach())
        return w2cs 
    
    def get_rkf_id(self):
        rkf_id_list = []
        for lm in self:
            for f in lm.frames[:-1]:
                id = f.time_idx + 2
            rkf_id_list.append(id)
        return rkf_id_list
    
    def get_gt_w2cs(self):
        gt_w2cs = []
        for lm in self:
            for f in lm.frames[:-1]:
                if f.time_idx == len(gt_w2cs) and lm.transform is not None:
                    gt_w2cs.append(torch.linalg.inv(f.gt_pose).detach())
        gt_w2cs.append(torch.linalg.inv(self[-1].frames[-1].gt_pose).detach())
        return gt_w2cs         
        