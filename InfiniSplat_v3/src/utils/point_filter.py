# Modified for SplatAnything: inference-only release.
"""Filter pixels near relative depth discontinuities before Gaussian merging."""
import torch
import torch.nn.functional as F


def flying_point_mask(depth, valid_mask=None, rtol=0.03, kernel_size=3):
    """Return a [B, S*H*W] keep mask, matching the source WorldGS depth-edge filter."""
    batch, views, height, width = depth.shape
    image = depth.reshape(-1, 1, height, width).float()
    padding = kernel_size // 2
    if valid_mask is None:
        upper = image
        lower = -image
    else:
        valid = valid_mask.reshape(-1, 1, height, width).bool()
        negative_inf = torch.full_like(image, float('-inf'))
        upper = torch.where(valid, image, negative_inf)
        lower = torch.where(valid, -image, negative_inf)
    spread = (F.max_pool2d(upper, kernel_size, stride=1, padding=padding)
              + F.max_pool2d(lower, kernel_size, stride=1, padding=padding))
    keep = ~((spread / image.clamp_min(1e-8)) > rtol)
    keep = keep.reshape(batch, views, height, width)
    if valid_mask is not None:
        keep &= valid_mask.bool()
    return keep.reshape(batch, -1)
