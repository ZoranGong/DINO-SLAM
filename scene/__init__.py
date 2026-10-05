# from .cameras_old import Cameras
from .Frame import Frame, LocalMap, Localmaps, Transform
from .Gaussians import Gaussians
import os
import json 
import torch 
import numpy as np

def save_scence(config: dict, gaussians: Gaussians, cameras: Localmaps, path):
    os.makedirs(path, exist_ok=True)
    
    config_path = os.path.join(path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    gaussians_path = os.path.join(path, 'gaussians.ply')
    gaussians.save_ply(gaussians_path)   
    
    w2cs_path = os.path.join(path, 'w2cs.npz')
    gt_w2cs_path = os.path.join(path, 'gt_w2cs.npz')

    w2cs = cameras.get_w2cs()
    gt_w2cs = cameras.get_gt_w2cs()
    w2cs = torch.stack(w2cs, dim=0).detach().cpu().numpy()
    gt_w2cs = torch.stack(gt_w2cs, dim=0).detach().cpu().numpy()
    np.save(w2cs_path, w2cs)
    np.save(gt_w2cs_path, gt_w2cs)
    

def load_scence(path):
    config_path = os.path.join(path, 'config.json')
    gaussians_path = os.path.join(path, 'gaussians.ply')
    w2cs_path = os.path.join(path, 'w2cs.npz.npy')
    gt_w2cs_path = os.path.join(path, 'gt_w2cs.npz.npy')
    
    assert os.path.exists(config_path) and os.path.exists(gaussians_path) and os.path.exists(w2cs_path) and os.path.exists(gt_w2cs_path), '''Scene file is broken !!!'''
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    gaussians = Gaussians(config).load_ply(gaussians_path)
    w2cs = torch.tensor(np.load(w2cs_path), dtype=torch.float32, device='cuda')
    gt_w2cs = torch.tensor(np.load(gt_w2cs_path), dtype=torch.float32, device='cuda')

    return config, gaussians, w2cs, gt_w2cs
    
    
        
    
    