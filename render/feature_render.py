"""
Rasterization of the per-Gaussian feature attribute with the 3-channel kernels:
the features are rendered in chunks of 3 channels through colors_precomp with the same geometry.
"""
import torch
from torch.utils.checkpoint import checkpoint

FEAT_CHUNK = 3


def feature_camera(rendercamera, scale):
    # the projection is resolution-invariant, only the raster size changes
    if scale == 1.0:
        return rendercamera
    return rendercamera._replace(image_height=max(16, int(round(rendercamera.image_height * scale))),
                                 image_width=max(16, int(round(rendercamera.image_width * scale))))


def _render_chunk(renderer, colors, means3D, means2D, opacities, scales, rotations, cov3D_precomp):
    return renderer(
        means3D,
        means2D,
        opacities=opacities,
        shs=None,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )[0]


def render_feature_map(renderer, means3D, means2D, opacities, scales, rotations, cov3D_precomp, features,
                       visible=None, use_checkpoint=False, detach_geometry=True):
    '''
    features        : [N, C] per-Gaussian features
    visible         : [N] bool mask of the Gaussians rasterized in the RGB pass (radii > 0)
    detach_geometry : feature losses update the features only, not means / scales / rotations / opacities
    return          : [C, H, W] feature map
    '''
    N, C = features.shape
    if detach_geometry:
        means3D, means2D, opacities = means3D.detach(), means2D.detach(), opacities.detach()
        scales = scales.detach() if scales is not None else None
        rotations = rotations.detach() if rotations is not None else None
        cov3D_precomp = cov3D_precomp.detach() if cov3D_precomp is not None else None

    if visible is not None and visible.sum() < N:
        idx = visible.nonzero(as_tuple=False).squeeze(1)
        means3D, means2D, opacities, features = means3D[idx], means2D[idx], opacities[idx], features[idx]
        scales = scales[idx] if scales is not None else None
        rotations = rotations[idx] if rotations is not None else None
        cov3D_precomp = cov3D_precomp[idx] if cov3D_precomp is not None else None
        N = idx.numel()

    pad = (-C) % FEAT_CHUNK
    if pad:
        features = torch.cat([features, features.new_zeros(N, pad)], dim=1)

    chunks = []
    for c0 in range(0, C + pad, FEAT_CHUNK):
        colors = features[:, c0:c0 + FEAT_CHUNK].contiguous()
        if use_checkpoint and torch.is_grad_enabled():
            out = checkpoint(_render_chunk, renderer, colors, means3D, means2D, opacities, scales, rotations,
                             cov3D_precomp, use_reentrant=False)
        else:
            out = _render_chunk(renderer, colors, means3D, means2D, opacities, scales, rotations, cov3D_precomp)
        chunks.append(out)
    return torch.cat(chunks, dim=0)[:C]
