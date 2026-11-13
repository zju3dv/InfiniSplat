# Modified for SplatAnything: inference-only release.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers.mlp import MlpFP32
from src.models.utils.grid import create_uv_grid, position_grid_to_embed

class DPTHead(nn.Module):
    """
    # DPT Head for dense prediction tasks.

    # This module implements the DPT (Dense Prediction Transformer) head as proposed in
    # "Vision Transformers for Dense Prediction" (https://arxiv.org/abs/2103.13413).
    # It takes features from a vision transformer backbone and generates dense (per-pixel) predictions
    # by fusing multi-scale features through a series of projection, upsampling, and refinement blocks.

    # Args:
    #   dim_in (int): Number of input feature channels.
    #   patch_size (int, optional): Patch size used by the backbone, default is 14.
    #   output_dim (int, optional): Number of output channels, default is 4.
    #   activation (str, optional): Activation function type for the output head, default is "inv_log".
    #   conf_activation (str, optional): Activation function type for the confidence/output uncertainty head, default is "expp1".
    #   features (int, optional): Number of channels used in intermediate feature representations, default is 256.
    #   out_channels (List[int], optional): Number of channels for each intermediate multi-scale feature.
    #   intermediate_layer_idx (List[int], optional): Indices specifying which backbone layers to use for multi-scale fusion.
    #   pos_embed (bool, optional): Whether to add positional encoding to the features, default is True.
    #   feature_only (bool, optional): If True, only return intermediate features (skip final prediction and activations).
    #   down_ratio (int, optional): Downsampling ratio of the output predictions, default is 1 (no downsampling).
    """

    def __init__(self, dim_in: int, patch_size: int=14, output_dim: int=4, activation: str='inv_log+expp1', features: int=256, out_channels: List[int]=[256, 512, 1024, 1024], pos_embed: bool=True, down_ratio: int=1) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.enable_depth_mask = True
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
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        conv2_in_channels = head_features_1 // 2
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0))

    def to(self, *args, **kwargs):
        self.norm = self.norm.to(*args, **kwargs)
        self.projects = self.projects.to(*args, **kwargs)
        self.resize_layers = self.resize_layers.to(*args, **kwargs)
        for key in ('layer1_rn', 'layer2_rn', 'layer3_rn', 'layer4_rn', 'refinenet1', 'refinenet2', 'refinenet3', 'refinenet4', 'output_conv1'):
            if not hasattr(self.scratch, key):
                continue
            setattr(self.scratch, key, getattr(self.scratch, key).to(*args, **kwargs))
        (args, kwargs) = MlpFP32.map_to_args_to_float(args, kwargs)
        self.scratch.output_conv2 = self.scratch.output_conv2.to(*args, **kwargs)
        return self

    def forward(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frames_chunk_size: int=8) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional frame chunking for memory efficiency."""
        (B, S, _, H, W) = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(token_list, images, patch_start_idx)
        assert frames_chunk_size > 0
        preds_chunks = []
        conf_chunks = []
        gs_chunks = []
        depth_mask_chunks = []
        for frame_start in range(0, S, frames_chunk_size):
            frame_end = min(frame_start + frames_chunk_size, S)
            (preds, conf, depth_mask) = self._forward_impl(token_list, images, patch_start_idx, frame_start, frame_end)
            preds_chunks.append(preds)
            conf_chunks.append(conf)
            depth_mask_chunks.append(depth_mask)
        return (torch.cat(preds_chunks, dim=1), torch.cat(conf_chunks, dim=1), torch.cat(depth_mask_chunks, dim=1))

    def _forward_impl(self, token_list: List[torch.Tensor], images: torch.Tensor, patch_start_idx: int, frame_start: int=None, frame_end: int=None) -> torch.Tensor:
        """Core forward implementation for DPT head."""
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
        fused = self.scratch_forward(feats)
        _interpolate_fn = lambda t: custom_interpolate(t, size=(int(ph * self.patch_size / self.down_ratio), int(pw * self.patch_size / self.down_ratio)), mode='bilinear', align_corners=True)
        fused = _interpolate_fn(fused)
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)
        out = self.scratch.output_conv2(fused.float().contiguous())
        (preds, conf, depth_mask) = self.activate_head(out, activation=self.activation)
        preds = preds.reshape(B, S, *preds.shape[1:])
        conf = conf.reshape(B, S, *conf.shape[1:])
        depth_mask = depth_mask.reshape(B, S, *depth_mask.shape[1:])
        return (preds, conf, depth_mask)

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float=0.1) -> torch.Tensor:
        """Apply positional embedding to tensor x."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed.to(x.dtype)

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion blocks."""
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

    def activate_head(self, out_head, activation=None):
        feat = out_head.permute(0, 2, 3, 1)
        return (feat[..., :-2].exp(), 1 + feat[..., -2].exp(), feat[..., -1])

def _make_fusion_block(features: int, size: int=None, has_residual: bool=True, groups: int=1) -> nn.Module:
    return FeatureFusionBlock(features, nn.ReLU(inplace=True), deconv=False, bn=False, expand=False, align_corners=True, size=size, has_residual=has_residual, groups=groups)

def _make_scratch(in_shape: List[int], out_shape: int, groups: int=1, expand: bool=False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

class ResidualConvUnit(nn.Module):
    """Residual convolution module with skip connection."""

    def __init__(self, features, activation, bn, groups=1):
        """Initialize ResidualConvUnit."""
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass with residual connection."""
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None, has_residual=True, groups=1):
        """Initialize FeatureFusionBlock."""
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass through the feature fusion block."""
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {'scale_factor': 2}
        elif size is None:
            modifier = {'size': self.size}
        else:
            modifier = {'size': size}
        output = custom_interpolate(output, **modifier, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def custom_interpolate(x: torch.Tensor, size: Tuple[int, int]=None, scale_factor: float=None, mode: str='bilinear', align_corners: bool=True) -> torch.Tensor:
    """Custom interpolation function to handle large tensors by chunking."""
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=input_elements // INT_MAX + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
