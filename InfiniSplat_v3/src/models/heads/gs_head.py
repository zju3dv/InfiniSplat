# Modified for SplatAnything: inference-only release.
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.heads.dense_head import _make_fusion_block, _make_scratch, custom_interpolate
from src.models.utils.grid import create_uv_grid, position_grid_to_embed

class GSFeatHead(nn.Module):
    """
    GS feature head that only outputs fused GS features.

    This head is used when gs depth is disabled. It skips the prediction
    conv (output_conv2) and returns only the fused GS feature map.
    """

    def __init__(self, dim_in: int, patch_size: int=14, features: int=256, out_channels: List[int]=[256, 512, 1024, 1024], pos_embed: bool=True, down_ratio: int=1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels])
        self.resize_layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0), nn.ConvTranspose2d(in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0), nn.Identity(), nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1)])
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        head_features_1 = features
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        conv2_in_channels = head_features_1 // 2
        self.input_merger = nn.Sequential(nn.Conv2d(3, conv2_in_channels, 7, 1, 3), nn.ReLU())

    def forward(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int=8, out_size: tuple=None) -> torch.Tensor:
        """Forward pass with optional frame chunking for memory efficiency."""
        (B, S, _, H, W) = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx, out_size=out_size)
        assert frames_chunk_size > 0
        gs_chunks = []
        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)
            gs = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end, out_size=out_size)
            gs_chunks.append(gs)
        return torch.cat(gs_chunks, dim=1)

    def forward_chunk(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frame_start: int, frame_end: int, out_size: tuple=None) -> torch.Tensor:
        """Compute the fused GS feature map for ONE frame range [frame_start, frame_end)."""
        return self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end, out_size=out_size)

    def _forward_impl(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frame_start: int=None, frame_end: int=None, out_size: tuple=None) -> torch.Tensor:
        if frame_start is not None and frame_end is not None:
            images = images[:, frame_start:frame_end].contiguous()
        (B, S, _, H, W) = images.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        feats = []
        for (proj, resize, tokens) in zip(self.projects, self.resize_layers, token_list):
            patch_tokens = tokens[:, :, patch_start_idx:]
            if frame_start is not None and frame_end is not None:
                patch_tokens = patch_tokens[:, frame_start:frame_end]
            patch_tokens = patch_tokens.reshape(B * S, -1, patch_tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)
            feat = patch_tokens.permute(0, 2, 1).reshape(B * S, patch_tokens.shape[-1], ph, pw)
            feat = proj(feat)
            if self.pos_embed:
                feat = self._apply_pos_embed(feat, W, H)
            feat = resize(feat)
            feats.append(feat)
        fused_hw = (int(ph * self.patch_size / self.down_ratio), int(pw * self.patch_size / self.down_ratio)) if out_size is None else (int(out_size[0]), int(out_size[1]))
        fused = self.scratch_forward(feats)
        _interpolate_fn = lambda t: custom_interpolate(t, size=fused_hw, mode='bilinear', align_corners=True)
        fused = _interpolate_fn(fused)
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)
        img_flat = images.reshape(B * S, -1, H, W)
        if (img_flat.shape[-2], img_flat.shape[-1]) != fused_hw:
            img_flat = F.interpolate(img_flat, size=fused_hw, mode='area')
        img_feat = self.input_merger(img_flat)
        fused = fused + img_feat
        fused = fused.reshape(B, S, *fused.shape[1:])
        return fused

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float=0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        (layer_1, layer_2, layer_3, layer_4) = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1
        out = self.scratch.output_conv1(out)
        return out
