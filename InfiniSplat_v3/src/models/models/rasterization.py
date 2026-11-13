# Modified for SplatAnything: inference-only release.
"""Inference decoding and fixed voxel merging (no rasterization or training)."""
import torch
import torch.nn as nn
from src.models.utils.geometry import depth_to_world_coords_points
from src.utils.point_filter import flying_point_mask

def da3_pose_align_factor(pred_c2w: torch.Tensor, ref_c2w: torch.Tensor) -> torch.Tensor:
    """Depth-alignment factor mapping PREDICTED-scale depth into the ref-camera scale.

    Ports da3_adapter._fallback_pose_scale (DA3's Umeyama fallback, used when the DA3 lib is
    unavailable — which it is here). DA3 extracts ONLY the Sim(3) SCALE from camera-center
    pairwise distances (orientation-invariant), then does depth / scale. Working on c2w
    centers directly, the per-sample factor is:

        factor = median_{i<j} ||C^ref_i - C^ref_j|| / ||C^pred_i - C^pred_j||

    so aligned_depth = pred_depth * factor lands in the ref-camera scale. Returns [B].
    S<2, degenerate predicted spread, or a zero-baseline reference (all ref cameras share a
    center, e.g. a cubemap/panorama) -> 1.0 (no rescale; keep the predicted depth scale).
    """
    pred_ctr = pred_c2w[..., :3, 3].float()                    # [B, S, 3]
    ref_ctr = ref_c2w[..., :3, 3].float()                      # [B, S, 3]
    B, n = pred_ctr.shape[0], pred_ctr.shape[1]
    if n < 2:
        return torch.ones(B, device=pred_ctr.device)
    i, j = torch.triu_indices(n, n, offset=1, device=pred_ctr.device)
    pred_d = (pred_ctr[:, i] - pred_ctr[:, j]).norm(dim=-1)    # [B, P]
    ref_d = (ref_ctr[:, i] - ref_ctr[:, j]).norm(dim=-1)       # [B, P]
    # A scale can only be read from pairs whose baseline is non-zero in BOTH frames. Excluding
    # ref_d~0 pairs makes this robust to zero/low-baseline reference cameras (e.g. a single-
    # viewpoint cubemap/panorama where every view shares one center): those pairs carry no scale
    # and would otherwise drag the median ratio to ~0, collapsing the whole cloud to the origin.
    # For normal spread-out multi-view captures all ref_d > 1e-6, so this is bit-exact.
    valid = (pred_d > 1.0e-6) & (ref_d > 1.0e-6)
    factor = torch.ones(B, device=pred_ctr.device)
    for b in range(B):
        vb = valid[b]
        if vb.any():
            factor[b] = torch.median(ref_d[b][vb] / pred_d[b][vb]).clamp_min(1.0e-6)
    return factor                                              # [B]

class DPTDecoder(nn.Module):
    def __init__(self, decode_chunk_size=8):
        super().__init__()
        self.decode_chunk_size = decode_chunk_size
        # Keep the original gs_renderer.decoder.gs_head.* checkpoint keys.
        self.gs_head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.ReLU(True),
            nn.Conv2d(256, 12, 1))

    def forward(self, features, images, depth, poses, intrinsics):
        batch, views, _, height, width = images.shape
        parts = []
        for start in range(0, views, self.decode_chunk_size):
            end = min(start + self.decode_chunk_size, views)
            feat = features(start, end).reshape(batch * (end-start), 128, height, width)
            parts.append(self.gs_head(feat.float()).reshape(batch, end-start, 12, height, width))
        raw = torch.cat(parts, dim=1).permute(0, 1, 3, 4, 2).reshape(batch, -1, 12)
        quats, scales, opacities, residual_sh, weights = raw.split([4, 3, 1, 3, 1], dim=-1)
        rgb = images.permute(0, 1, 3, 4, 2).reshape(batch, -1, 3)
        means, _, _ = depth_to_world_coords_points(
            depth.reshape(batch*views, height, width), poses.reshape(batch*views, 4, 4),
            intrinsics.reshape(batch*views, 3, 3))
        return dict(means=means.reshape(batch, -1, 3), base_rgb=rgb,
                    quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
                    scales=scales.exp().clamp_max(0.3), opacities=opacities[..., 0].sigmoid(),
                    sh=((rgb - 0.5) / 0.28209479177387814 + residual_sh).unsqueeze(-2),
                    weights=weights[..., 0].sigmoid())


class GaussianSplatRenderer(nn.Module):
    def __init__(self, use_gt_camera=False, spacing_target=0.0013227513,
                 voxel_size=0.0018, max_scale_pct=99.8, decode_chunk_size=8,
                 filter_flying_points_infer=False, filter_sky_infer=False,
                 filter_large_scale_infer=True, max_scale_abs=None):
        super().__init__()
        self.use_gt_camera = use_gt_camera
        self.spacing_target = spacing_target
        self.voxel_size = voxel_size
        self.max_scale_pct = max_scale_pct
        self.max_scale_abs = max_scale_abs
        self.filter_flying_points_infer = filter_flying_points_infer
        self.filter_sky_infer = filter_sky_infer
        self.filter_large_scale_infer = filter_large_scale_infer
        self.decoder = DPTDecoder(decode_chunk_size)

    def normalize_gaussian_spacing(self, depth, poses, intrinsics, predicted_poses,
                                   valid_mask=None):
        """Gaussian Spacing Normalization (GSN): SplatAnything's scene-scale transform.

        First reconcile predicted depth with the selected camera translation scale.
        Then normalize each scene's median per-pixel spacing, depth / sqrt(fx * fy),
        to spacing_target. This fronto-parallel footprint defines a consistent scale
        for Gaussian decoding and voxel merging across scenes and focal lengths.

        One factor per scene scales all depths and camera translations together;
        rotations and intrinsics are preserved. The optional non-sky mask affects
        only the spacing statistic, not which pixels are returned.

        Args:
            depth: Predicted depth, [B, S, H, W].
            poses: Selected OpenCV camera-to-world matrices, [B, S, 4, 4].
            intrinsics: Pixel intrinsics at the processing resolution, [B, S, 3, 3].
            predicted_poses: Geometry-model camera-to-world matrices, [B, S, 4, 4].
            valid_mask: Optional boolean [B, S, H, W] mask for the scale statistic.

        Returns:
            Normalized depth and camera-to-world matrices, without modifying inputs.
            If a scene has no valid samples, its aligned depth scale is retained.
        """
        batch = depth.shape[0]
        align = da3_pose_align_factor(predicted_poses, poses)
        depth = depth * align[:, None, None, None]
        focal = (intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).clamp_min(0).sqrt().clamp_min(1e-6)
        factor = torch.ones(batch, device=depth.device)
        for b in range(batch):
            spacing = depth[b] / focal[b, :, None, None]
            valid = (depth[b] > 0) & torch.isfinite(spacing)
            if valid_mask is not None:
                valid &= valid_mask[b]
            if valid.any():
                factor[b] = self.spacing_target / spacing[valid].median().clamp_min(1e-6)
        depth = depth * factor[:, None, None, None]
        poses = poses.clone()
        poses[..., :3, 3] *= factor[:, None, None]
        return depth, poses

    def forward(self, views, geometry, features):
        images = views['img']
        batch, count, _, height, width = images.shape
        depth = geometry['depth'].reshape(batch, count, height, width)
        cameras = views if self.use_gt_camera else geometry
        poses, intrinsics = cameras['camera_poses'], cameras['camera_intrs']
        sky_keep = views.get('depthmap') if self.filter_sky_infer else None
        sky_keep = sky_keep >= 0 if sky_keep is not None else None

        # Gaussian Spacing Normalization (GSN) establishes the scene scale before decoding.
        depth, poses = self.normalize_gaussian_spacing(
            depth, poses, intrinsics, geometry['camera_poses'], valid_mask=sky_keep)
        splats = self.decoder(features, images, depth, poses, intrinsics)

        # Apply all enabled masks before point-cloud export and voxel merging.
        scale = splats['scales'].amax(dim=-1)
        keep = torch.ones_like(scale, dtype=torch.bool)
        if self.filter_large_scale_infer:
            if self.max_scale_abs is not None:
                keep &= scale <= self.max_scale_abs
            if self.max_scale_pct is not None:
                for b in range(batch):
                    n = scale.shape[1]
                    k = min(max(round(self.max_scale_pct / 100 * n), 1), n)
                    keep[b] &= scale[b] <= scale[b].float().kthvalue(k).values
        if self.filter_flying_points_infer:
            keep &= flying_point_mask(depth, valid_mask=views.get('valid_mask'))
        if sky_keep is not None:
            keep &= sky_keep.reshape(batch, -1)
        result = dict(geometry)
        result['points_ply_means'] = [splats['means'][b][keep[b]] for b in range(batch)]
        result['points_ply_rgb'] = [splats['base_rgb'][b][keep[b]] for b in range(batch)]
        result['splats'] = self.merge_voxels(splats, keep)
        # Save cameras in the same canonical coordinate frame as the exported cloud.
        result['export_camera_poses'] = poses
        result['export_camera_intrs'] = intrinsics
        return result

    def merge_voxels(self, splats, keep):
        output = {k: [] for k in ('means', 'quats', 'scales', 'opacities', 'sh')}
        for b in range(splats['means'].shape[0]):
            scene = {k: v[b][keep[b]] for k, v in splats.items()}
            if not len(scene['means']):
                for k in output:
                    output[k].append(scene[k])
                continue
            coords = (scene['means'] / self.voxel_size).floor().long()
            coords = coords - coords.min(dim=0).values
            dims = coords.max(dim=0).values + 1
            keys = coords[:, 0] * dims[1] * dims[2] + coords[:, 1] * dims[2] + coords[:, 2]
            unique, inverse = torch.unique(keys, return_inverse=True)
            weights = scene['weights']
            sums = weights.new_zeros(len(unique)).scatter_add_(0, inverse, weights).clamp_min(1e-8)
            for name in output:
                # Preserve the trained configuration's merge rule: opacity = sum(w*w)/sum(w).
                values = weights if name == 'opacities' else scene[name]
                shape = values.shape[1:]
                merged = values.new_zeros((len(unique), *shape))
                weight_shape = (-1,) + (1,) * len(shape)
                for column in range(values.reshape(len(values), -1).shape[1]):
                    merged.reshape(len(unique), -1)[:, column].scatter_add_(
                        0, inverse, values.reshape(len(values), -1)[:, column] * weights)
                if name == 'quats':
                    merged /= merged.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                else:
                    merged /= sums.reshape(weight_shape)
                output[name].append(merged)
        return output
