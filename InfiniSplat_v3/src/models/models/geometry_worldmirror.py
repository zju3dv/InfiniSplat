# Modified for SplatAnything: inference-only release.
"""WorldMirror-2 depth and camera predictions for Gaussian inference."""
import torch
from src.models.heads.camera_head import CameraHead
from src.models.heads.dense_head import DPTHead
from src.models.models.worldmirror_base import WorldMirrorEncoderBase

class GeometryWorldMirror(WorldMirrorEncoderBase):
    def __init__(self, **encoder_kwargs):
        super().__init__(**encoder_kwargs)
        self.enable_depth_mask = True
        self.cam_head = CameraHead(dim_in=2048)
        self.depth_head = DPTHead(dim_in=2048, output_dim=3, patch_size=14,
                                  activation='exp+expp1+linear')

    @torch.no_grad()
    def forward(self, views, cond_flags=(0, 0, 0)):
        tokens, patch_start = self.encode(views, cond_flags)
        height, width = views['img'].shape[-2:]
        params = self.cam_head(tokens)[-1]
        poses, intrinsics = self.transform_camera_vector(params, height, width)
        depth, confidence, mask_logits = self.depth_head(
            tokens, images=views['img'], patch_start_idx=patch_start)
        return dict(camera_params=params, camera_poses=poses, camera_intrs=intrinsics,
                    depth=depth, depth_conf=confidence, depth_mask_logits=mask_logits)
