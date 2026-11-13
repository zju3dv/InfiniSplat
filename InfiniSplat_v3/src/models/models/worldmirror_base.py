# Modified for SplatAnything: inference-only release.
"""Shared WorldMirror-2 encoder used by both inference branches."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.models.visual_transformer import VisualGeometryTransformer
from src.models.utils.camera_utils import extrinsics_to_vector, vector_to_camera_matrices
from src.models.utils.priors import normalize_depth, normalize_poses

class WorldMirrorEncoderBase(nn.Module):
    def __init__(self, enable_cond=False, normalized_rope=True,
                 rope_normalize_coords='separate', rope_base=100.0):
        super().__init__()
        self.enable_cond = enable_cond
        self.enable_bf16 = False
        self.visual_geometry_transformer = VisualGeometryTransformer(
            img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
            patch_embed='dinov2_vitl14_reg', enable_cond=enable_cond,
            normalized_rope=normalized_rope, rope_normalize_coords=rope_normalize_coords,
            rope_base=rope_base,
        )

    def encode(self, views, cond_flags):
        priors = self.extract_priors(views) if any(cond_flags) else None
        return self.visual_geometry_transformer(views['img'], priors, cond_flags=cond_flags)

    def extract_priors(self, views):
        """Extract and normalize the optional geometric conditioning priors.

        Returns:
            (depths, rays, poses), each [B, S, ...] or None when the input lacks that prior.
        """
        h, w = views["img"].shape[-2:]
        depths = rays = poses = None

        if "camera_poses" in views:
            extrinsics = views["camera_poses"][:, :, :3]  # c2w
            extrinsics = normalize_poses(extrinsics)
            cam_params = extrinsics_to_vector(extrinsics)
            poses = cam_params[:, :, :7]                  # [B, S, 7]
            if self.enable_bf16:
                poses = poses.to(torch.bfloat16)

        if "depthmap" in views:
            depth_h, depth_w = views["depthmap"].shape[-2:]
            depths = views["depthmap"]
            if depth_h != h or depth_w != w:
                depths = F.interpolate(
                    depths, size=(h, w), mode="bilinear", align_corners=False
                )
            depths = normalize_depth(depths)               # [B, S, H, W]
            if self.enable_bf16:
                depths = depths.to(torch.bfloat16)

        if "camera_intrs" in views:
            intrinsics = views["camera_intrs"][:, :, :3, :3]
            fx, fy = intrinsics[:, :, 0, 0] / w, intrinsics[:, :, 1, 1] / h
            cx, cy = intrinsics[:, :, 0, 2] / w, intrinsics[:, :, 1, 2] / h
            rays = torch.stack([fx, fy, cx, cy], dim=-1)   # [B, S, 4]
            if self.enable_bf16:
                rays = rays.to(torch.bfloat16)

        return depths, rays, poses

    @staticmethod
    def transform_camera_vector(camera_params, h, w):
        """Camera parameter vector -> (c2w [B, S, 4, 4], intrinsics [B, S, 3, 3])."""
        ext_mat, int_mat = vector_to_camera_matrices(camera_params, image_hw=(h, w))
        homo_row = torch.tensor(
            [0, 0, 0, 1], device=ext_mat.device, dtype=ext_mat.dtype
        ).view(1, 1, 1, 4)
        homo_row = homo_row.repeat(ext_mat.shape[0], ext_mat.shape[1], 1, 1)
        w2c_mat = torch.cat([ext_mat, homo_row], dim=2)
        try:
            c2w_mat = torch.linalg.inv(w2c_mat)
        except Exception:
            # cusolver occasionally fails on near-singular batches; CPU inverse is the fallback.
            c2w_mat = torch.linalg.inv(w2c_mat.cpu()).to(camera_params.device)
        return c2w_mat, int_mat
