"""
Scene Geometry Encoder (SGE) and GeoDINO features.
"""
import os
import threading
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MLP(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c_in, c_hidden), nn.GELU(), nn.Linear(c_hidden, c_out))

    def forward(self, x):
        return self.net(x)


def sine_pos_embed(hp, wp, d, device):
    assert d % 4 == 0
    d4 = d // 4
    omega = torch.arange(d4, device=device, dtype=torch.float32) / d4
    omega = 1.0 / (10000 ** omega)
    ys, xs = torch.meshgrid(torch.arange(hp, device=device, dtype=torch.float32),
                            torch.arange(wp, device=device, dtype=torch.float32), indexing='ij')
    ys = ys.reshape(-1, 1) * omega[None]
    xs = xs.reshape(-1, 1) * omega[None]
    return torch.cat([ys.sin(), ys.cos(), xs.sin(), xs.cos()], dim=1)


class DummyDINO(nn.Module):
    def __init__(self, patch_size=14, embed_dim=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g) / patch_size)

    def forward_features(self, x):
        t = torch.tanh(self.proj(x))
        return {"x_norm_patchtokens": t.flatten(2).transpose(1, 2)}


class DinoBackbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', input_height=448, weights=None, hub_repo='facebookresearch/dinov2'):
        super().__init__()
        if name == 'dummy':
            self.model = DummyDINO()
        else:
            self.model = self._load_hub(hub_repo, name, weights)
        self.patch = int(self.model.patch_size)
        self.dim = int(self.model.embed_dim)
        self.input_height = int(round(input_height / self.patch)) * self.patch
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _load_hub(repo, name, weights):
        if weights is not None and os.path.isfile(weights):
            model = torch.hub.load(repo, name, source='github', pretrained=False, trust_repo=True, skip_validation=True)
            model.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)
            return model
        try:
            return torch.hub.load(repo, name, source='github', pretrained=True, trust_repo=True, skip_validation=True)
        except Exception as e:
            raise RuntimeError(f"Could not load DINOv2 '{name}' from torch.hub ({e}). "
                               f"Set sge.dino_weights to a local checkpoint or sge.dino_model='dummy'.") from e

    def train(self, mode=True):
        return super().train(False)

    def target_size(self, H, W):
        hin = self.input_height
        win = max(self.patch, int(round(W * hin / H / self.patch)) * self.patch)
        return hin, win

    @torch.no_grad()
    def forward(self, rgb):
        H, W, _ = rgb.shape
        hin, win = self.target_size(H, W)
        x = rgb.permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(hin, win), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        hp, wp = hin // self.patch, win // self.patch
        return tokens[0].transpose(0, 1).reshape(self.dim, hp, wp).contiguous()


class SceneGeometryEncoder(nn.Module):
    def __init__(self, dino_dim, d_model=128, num_heads=4, patch=14, depth_scale=5.0):
        super().__init__()
        self.d = d_model
        self.patch = patch
        self.depth_scale = depth_scale
        self.mlp_a = MLP(dino_dim, d_model, d_model)
        self.depth_patch = nn.Conv2d(2, d_model, kernel_size=patch, stride=patch)
        self.mlp_g = MLP(d_model, d_model, d_model)
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_k1 = nn.LayerNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, num_heads, kdim=d_model, vdim=dino_dim, batch_first=True)
        self.norm_q2 = nn.LayerNorm(d_model)
        self.norm_k2 = nn.LayerNorm(d_model)
        self.attn2 = nn.MultiheadAttention(d_model, num_heads, kdim=d_model, vdim=d_model, batch_first=True)
        self.mlp_post = MLP(d_model, d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def depth_tokens(self, depth, hp, wp):
        d = depth.reshape(1, 1, depth.shape[0], depth.shape[1]).float()
        valid = (d > 1e-5).float()
        d = F.interpolate(d, size=(hp * self.patch, wp * self.patch), mode='nearest')
        valid = F.interpolate(valid, size=(hp * self.patch, wp * self.patch), mode='nearest')
        d = torch.clamp(d / self.depth_scale, 0.0, 4.0) * valid
        f_g = self.depth_patch(torch.cat([d, valid], dim=1))
        return f_g.flatten(2).transpose(1, 2)

    def forward(self, f_dino, depth):
        C, hp, wp = f_dino.shape
        tok_dino = f_dino.flatten(1).transpose(0, 1)[None]
        pos = sine_pos_embed(hp, wp, self.d, f_dino.device)[None]

        f_a = self.mlp_a(tok_dino) + pos
        f_g = self.mlp_g(self.depth_tokens(depth, hp, wp)) + pos

        f_atten_d, _ = self.attn1(self.norm_q1(f_g), self.norm_k1(f_a), tok_dino, need_weights=False)
        f_atten_a, _ = self.attn2(self.norm_q2(f_g), self.norm_k2(f_atten_d), f_a, need_weights=False)
        f_gD = self.norm_out(f_g + self.mlp_post(f_atten_a) + f_atten_d)
        return f_gD[0].transpose(0, 1).reshape(self.d, hp, wp)


class GeoDINOHead(nn.Module):
    def __init__(self, d_model, dino_dim, feat_dim=16, hidden=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.mlp_reduce = MLP(d_model, hidden, feat_dim)
        self.mlp_upsample = MLP(feat_dim, hidden, dino_dim)

    def reduce(self, f_gD):
        d, h, w = f_gD.shape
        return self.mlp_reduce(f_gD.flatten(1).transpose(0, 1)).transpose(0, 1).reshape(self.feat_dim, h, w)

    def upsample(self, feat_map):
        c, h, w = feat_map.shape
        out = self.mlp_upsample(feat_map.flatten(1).transpose(0, 1))
        return out.transpose(0, 1).reshape(-1, h, w)


class SGESystem:
    STAGE1, WARMUP, JOINT = 'stage1', 'warmup', 'joint'

    def __init__(self, config: dict, trainable: bool = True, device='cuda'):
        cfg = config['sge']
        self.device = device
        self.feat_dim = int(cfg.get('feat_dim', 16))
        self.loss_res_scale = int(cfg.get('loss_res_scale', 2))
        self.alpha_th = float(cfg.get('mask_alpha_th', 0.5))
        self.use_amp = bool(cfg.get('dino_fp16', True))

        self.backbone = DinoBackbone(cfg.get('dino_model', 'dinov2_vitb14'),
                                     input_height=cfg.get('dino_input_height', 448),
                                     weights=cfg.get('dino_weights', None)).to(device)
        self.encoder = SceneGeometryEncoder(self.backbone.dim,
                                            d_model=int(cfg.get('d_model', 128)),
                                            num_heads=int(cfg.get('num_heads', 4)),
                                            patch=self.backbone.patch,
                                            depth_scale=float(cfg.get('depth_scale', 5.0))).to(device)
        self.head = GeoDINOHead(self.encoder.d, self.backbone.dim, self.feat_dim,
                                hidden=int(cfg.get('head_hidden', 128))).to(device)

        self.n_stage1 = int(cfg.get('N_stage1', config.get('backend', {}).get('num_ba_iters', 0)))
        self.n_warmup = int(cfg.get('N_warmup', 20))
        self.n_joint = int(cfg.get('N_joint', -1))
        self.sge_iter = 0

        self.lr_joint = float(cfg.get('lr', 1e-3))
        self.lr_warmup = float(cfg.get('warmup_lr', self.lr_joint))
        self.optimizer = None
        if trainable:
            self.optimizer = torch.optim.Adam(self.trainable_parameters(), lr=self.lr_warmup, eps=1e-8)
            self.encoder.train()
            self.head.train()
        else:
            for p in self.trainable_parameters():
                p.requires_grad_(False)
            self.encoder.eval()
            self.head.eval()

        # CPU fp16 cache of frozen DINO features
        self._dino_cache = OrderedDict()
        self._dino_cache_size = int(cfg.get('dino_cache_size', 16))
        self._cache_lock = threading.Lock()
        self._prefetch_thread = None

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def state_dict(self):
        return {'encoder': self.encoder.state_dict(), 'head': self.head.state_dict(), 'sge_iter': self.sge_iter}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def stage(self, mapping_iter: int) -> str:
        if mapping_iter < self.n_stage1:
            return self.STAGE1
        if self.sge_iter < self.n_warmup:
            return self.WARMUP
        if self.n_joint >= 0 and self.sge_iter >= self.n_warmup + self.n_joint:
            return self.STAGE1
        return self.JOINT

    def step_counter(self):
        self.sge_iter += 1

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def optimizer_step(self):
        if self.optimizer is None:
            return
        lr = self.lr_warmup if self.sge_iter < self.n_warmup else self.lr_joint
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.step()

    def dino(self, key, rgb):
        with self._cache_lock:
            if key is not None and key in self._dino_cache:
                self._dino_cache.move_to_end(key)
                return self._dino_cache[key].to(self.device, non_blocking=True).float()
        f = self._run_backbone(rgb)
        self._cache_put(key, f)
        return f

    def _run_backbone(self, rgb):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(self.device != 'cpu' and self.use_amp)):
            f = self.backbone(rgb.to(self.device))
        return f.float()

    def _cache_put(self, key, f):
        if key is None:
            return
        with self._cache_lock:
            self._dino_cache[key] = f.detach().to('cpu', dtype=torch.float16)
            while len(self._dino_cache) > self._dino_cache_size:
                self._dino_cache.popitem(last=False)

    def prefetch(self, items):
        items = [(k, rgb) for k, rgb in items if rgb is not None]
        if not items:
            return
        prev = self._prefetch_thread

        def _work():
            if prev is not None:
                prev.join()
            stream = torch.cuda.Stream() if self.device != 'cpu' else None
            for k, rgb in items:
                with self._cache_lock:
                    cached = k in self._dino_cache
                if cached:
                    continue
                with torch.no_grad():
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            f = self._run_backbone(rgb)
                        stream.synchronize()
                    else:
                        f = self._run_backbone(rgb)
                self._cache_put(k, f)

        self._prefetch_thread = threading.Thread(target=_work, daemon=True)
        self._prefetch_thread.start()

    def encode16(self, f_dino, depth):
        return self.head.reduce(self.encoder(f_dino, depth.to(self.device)))

    @torch.no_grad()
    def init_feature_map(self, rgb, depth, key=None):
        H, W = rgb.shape[0], rgb.shape[1]
        f16 = self.encode16(self.dino(key, rgb), depth)
        f16 = F.interpolate(f16[None], size=(H, W), mode='bilinear', align_corners=False)[0]
        return f16.permute(1, 2, 0).contiguous()

    def losses(self, key, rgb, depth, render_pkg: dict) -> dict:
        '''
        feat         : L1(mlp_upsample(rendered feat), f_dino)
        feat_consist : L1(rendered feat, SGE feat)
        feat_ae      : L1(mlp_upsample(SGE feat), f_dino)
        '''
        f_dino = self.dino(key, rgb)
        f16_sge = self.encode16(f_dino, depth)
        _, hp, wp = f_dino.shape
        hl, wl = hp * self.loss_res_scale, wp * self.loss_res_scale

        losses = {}
        losses['feat_ae'] = F.l1_loss(self.head.upsample(f16_sge), f_dino)

        render_feat = render_pkg.get('render_feat', None)
        if render_feat is None:
            return losses
        render_feat = torch.nan_to_num(render_feat, 0, 0)
        r16 = F.interpolate(render_feat[None], size=(hl, wl), mode='area')[0]
        alpha = F.interpolate(torch.nan_to_num(render_pkg['render_alpha'], 0, 0).detach()[None],
                              size=(hl, wl), mode='area')[0]
        mask = (alpha > self.alpha_th).float()
        denom = mask.sum().clamp_min(1.0)

        f_dino_l = F.interpolate(f_dino[None], size=(hl, wl), mode='bilinear', align_corners=False)[0]
        err = (self.head.upsample(r16) - f_dino_l).abs().mean(dim=0, keepdim=True)
        losses['feat'] = (err * mask).sum() / denom

        f16_l = F.interpolate(f16_sge[None], size=(hl, wl), mode='bilinear', align_corners=False)[0]
        err_c = (r16 - f16_l).abs().mean(dim=0, keepdim=True)
        losses['feat_consist'] = (err_c * mask).sum() / denom
        return losses


def sge_enabled(config: dict) -> bool:
    return bool(config.get('sge', {}).get('enable', False))
