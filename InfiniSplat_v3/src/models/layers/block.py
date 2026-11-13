# Modified for SplatAnything: inference-only release.
from typing import Callable
import torch
from torch import nn, Tensor
from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

class Block(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4.0, qkv_bias: bool=True, proj_bias: bool=True, ffn_bias: bool=True, drop: float=0.0, attn_drop: float=0.0, init_values=None, drop_path: float=0.0, act_layer: Callable[..., nn.Module]=nn.GELU, norm_layer: Callable[..., nn.Module]=nn.LayerNorm, attn_class: Callable[..., nn.Module]=Attention, ffn_layer: Callable[..., nn.Module]=Mlp, qk_norm: bool=False, fused_attn: bool=True, rope=None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, qk_norm=qk_norm, fused_attn=fused_attn, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:

        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        x = x + attn_residual_func(x, pos=pos)
        x = x + ffn_residual_func(x)
        return x

class NestedTensorBlock(Block):
    pass
