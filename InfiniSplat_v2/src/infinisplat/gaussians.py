"""Native RDNorm direct-scale decoding and uniform voxel merge (SH0)."""
import torch
from torch import nn

SH_C0 = 0.28209479177387814

class GaussianHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, bias=False),
                                    nn.ReLU(True), nn.Conv2d(256, 11, 1))

    def forward(self, features, images, depth, c2w, intrinsics):
        features = features.float()
        raw = torch.cat([self.layers(frame[None]) for frame in features.flatten(0, 1)], 0)
        raw = raw.permute(0, 2, 3, 1).reshape(-1, 11)
        rotation, scale, opacity, sh = torch.split(raw, [4, 3, 1, 3], -1)
        rgb = images[0].permute(0, 2, 3, 1).reshape(-1, 3).float()
        return dict(quats=rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-8),
                    scales=scale.exp().clamp_max(.3), opacities=opacity[:, 0].sigmoid(),
                    sh=(((rgb - .5) / SH_C0).to(sh.dtype) + sh)[:, None], base_rgb=rgb)

@torch.no_grad()
def merge(splats, *, return_membership=False):
    """Median uses lower-middle torch.median; no quantile approximation or cap."""
    keep = torch.ones(len(splats['means']), dtype=torch.bool, device=splats['means'].device)
    for key in ('means','quats','scales','opacities','sh'):
        keep &= torch.isfinite(splats[key].flatten(1) if splats[key].ndim > 1 else splats[key][:, None]).all(-1)
    keep &= (splats['scales'] > 0).all(-1) & (splats['quats'].norm(dim=-1) > 1e-8)
    if not keep.any():
        raise ValueError('No valid Gaussians')
    if not torch.isfinite(splats['scales']).all() or not (splats['scales'] > 0).all():
        raise ValueError('Invalid direct scale population')
    median = splats['scales'].float().log().mean(-1).exp()[keep].median()
    voxel = median * 1.4
    s = {k:v[keep].float() for k,v in splats.items() if k != 'base_rgb'}
    grid = (s['means'] / float(voxel)).floor().long()
    grid -= grid.min(0).values
    dims = grid.max(0).values + 1
    # Match native voxel order; guard int64 overflow instead of aliasing cells.
    if int(dims[0]) * int(dims[1]) * int(dims[2]) > torch.iinfo(torch.int64).max:
        raise ValueError('Scene extent exceeds the native voxel indexing range')
    flat = grid[:, 0] * dims[1] * dims[2] + grid[:, 1] * dims[2] + grid[:, 2]
    unique, inverse = torch.unique(flat, return_inverse=True)
    counts = torch.zeros(len(unique), device=grid.device)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    merged = {}
    for key, value in s.items():
        shape = value.shape[1:]
        columns = value.reshape(len(value), -1)
        result = columns.new_zeros((len(unique), columns.shape[1]))
        for column in range(columns.shape[1]):
            result[:, column].scatter_add_(0, inverse, columns[:, column])
        merged[key] = (result / counts[:, None]).reshape(len(unique), *shape)
    merged['quats'] /= merged['quats'].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    visible = merged['opacities'] >= .01
    result = {k:v[visible] for k,v in merged.items()}
    if not visible.any():
        raise ValueError('No Gaussians remain after native opacity pruning')
    stats = dict(raw_count=len(keep), valid_count=int(keep.sum()),
                        merged_count=len(unique), final_count=int(visible.sum()),
                        median_scale=float(median), voxel_size=float(voxel),
                        voxel_multiplier=1.4, opacity_threshold=.01)
    if return_membership:
        return result, stats, dict(keep=keep, inverse=inverse, counts=counts, visible=visible)
    return result, stats
