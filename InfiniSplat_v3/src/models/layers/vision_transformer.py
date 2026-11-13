# Modified for SplatAnything: inference-only release.
from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from . import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
log = logging.getLogger(__name__)

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for (child_name, child_module) in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class BlockChunk(nn.ModuleList):

    def forward(self, x):
        for b in self:
            x = b(x)
        return x

class DinoVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True, proj_bias=True, drop_path_rate=0.0, drop_path_uniform=False, init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU, block_fn=Block, ffn_layer='mlp', block_chunks=1, num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1, qk_norm=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_reentrant = False
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        if ffn_layer == 'mlp':
            log.info('using MLP layer as FFN')
            ffn_layer = Mlp
        elif ffn_layer == 'swiglufused' or ffn_layer == 'swiglu':
            log.info('using SwiGLU layer as FFN')
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == 'identity':
            log.info('using Identity layer as FFN')

            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError
        blocks_list = [block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer, init_values=init_values, qk_norm=qk_norm) for i in range(depth)]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i:i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-06)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-06)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs['scale_factor'] = (sx, sy)
        else:
            kwargs['size'] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode='bicubic', antialias=self.interpolate_antialias, **kwargs)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        (B, nc, w, h) = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {'x_norm_clstoken': x_norm[:, 0], 'x_norm_regtokens': x_norm[:, 1:self.num_register_tokens + 1], 'x_norm_patchtokens': x_norm[:, self.num_register_tokens + 1:], 'x_prenorm': x, 'masks': masks}

    def forward(self, x):
        return self.forward_features(x)

def init_weights_vit_timm(module: nn.Module, name: str=''):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention), num_register_tokens=num_register_tokens, **kwargs)
    return model
