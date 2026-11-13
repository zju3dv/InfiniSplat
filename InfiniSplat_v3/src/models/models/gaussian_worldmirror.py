# Modified for SplatAnything: inference-only release.
"""Stage-1 Gaussian student with the release DPT decoder."""
import torch
from src.models.heads.gs_head import GSFeatHead
from src.models.models.worldmirror_base import WorldMirrorEncoderBase
from src.models.models.rasterization import GaussianSplatRenderer

class GaussianWorldMirror(WorldMirrorEncoderBase):
    def __init__(self, use_gt_camera=False, spacing_target=0.0013227513,
                 gs_voxel_size=0.0018, max_scale_pct=99.8, decode_chunk_size=8,
                 filter_flying_points_infer=False, filter_sky_infer=False,
                 filter_large_scale_infer=True, max_scale_abs=None,
                 **encoder_kwargs):
        super().__init__(**encoder_kwargs)
        self.gs_head = GSFeatHead(dim_in=2048, patch_size=14, features=256)
        self.gs_renderer = GaussianSplatRenderer(
            use_gt_camera=use_gt_camera, spacing_target=spacing_target,
            voxel_size=gs_voxel_size, max_scale_pct=max_scale_pct,
            decode_chunk_size=decode_chunk_size,
            filter_flying_points_infer=filter_flying_points_infer,
            filter_sky_infer=filter_sky_infer,
            filter_large_scale_infer=filter_large_scale_infer,
            max_scale_abs=max_scale_abs)

    @torch.no_grad()
    def forward(self, views, geometry, cond_flags=(0, 0, 0)):
        tokens, patch_start = self.encode(views, cond_flags)
        def features(start, end):
            return self.gs_head.forward_chunk(tokens, views['img'], patch_start, start, end)
        # The dense decoder and Gaussian activations use FP32, as in the source configuration.
        with torch.autocast(device_type=views['img'].device.type, enabled=False):
            return self.gs_renderer(views, geometry, features)
