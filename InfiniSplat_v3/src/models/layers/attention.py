# Modified for SplatAnything: inference-only release.
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, qkv_bias: bool=True, proj_bias: bool=True, attn_drop: float=0.0, proj_drop: float=0.0, norm_layer: nn.Module=nn.LayerNorm, qk_norm: bool=False, fused_attn: bool=True, rope=None) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        (q, k) = (self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype))
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
