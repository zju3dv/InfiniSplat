"""InfiniSplat v2: DA3 features, Gaussian decoding and RDNorm fusion."""
import torch
from torch import nn
from .gaussians import GaussianHead, merge
from .decoder import GSFeatHead, depth_only
from .geometry import normalize_predicted_point_radius, align_rdnorm_gt_via_legacy_da3
from .geometry import normalize_all_camera_poses

class InfiniSplat(nn.Module):
    def __init__(self, backbone_config, appearance_features_only=False):
        super().__init__()
        self.appearance_features_only = appearance_features_only
        self.gs_feature_head = GSFeatHead(2048)
        self.gs_feature_head.input_merger = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.GELU(), nn.Conv2d(32,64,3,padding=1),
            nn.GELU(), nn.Conv2d(64,128,3,padding=1), nn.GELU())
        self.gaussian_head = GaussianHead()
        # Always use the torch SwiGLU implementation: no xformers dependency.
        import depth_anything_3.model.dinov2.vision_transformer as da3_vit
        from .swiglu import SwiGLUFFNFused
        da3_vit.SwiGLUFFNFused = SwiGLUFFNFused
        from .da3_order import _get_intermediate_layers_not_chunked
        da3_vit.DinoVisionTransformer._get_intermediate_layers_not_chunked = _get_intermediate_layers_not_chunked
        from depth_anything_3.cfg import create_object
        from omegaconf import OmegaConf
        # Use the upstream network only; no image/video/point-cloud API imports.
        self.appearance = create_object(OmegaConf.create(backbone_config))
        self.geometry = create_object(OmegaConf.create(backbone_config))
        from depth_anything_3.model.dualdpt import DualDPT
        if appearance_features_only:
            # Appearance contributes tokens only; depth and camera outputs are unused.
            self.appearance.head = nn.Identity()
            self.appearance.cam_dec = None
        import types
        def depth_head(branch, feats, H, W):
            if branch is self.geometry and type(branch.head) is DualDPT and branch.cam_dec is not None:
                return depth_only(branch.head, feats, H, W)
            return branch.head(feats,H,W,patch_start_idx=0,chunk_size=1)
        for branch in (self.appearance,self.geometry):
            branch._process_depth_head=types.MethodType(depth_head,branch)
        self.projections = nn.ModuleList([nn.Linear(1536,2048) for _ in range(4)])

    def configure_inference(self):
        """Reuse features only when both branches have identical loaded tensors.

        Both branches are constructed from the same backbone config. Check every
        parameter and buffer after strict weight loading; independent branch
        weights keep the original two-backbone path.
        """
        def equal(left, right):
            if left is None or right is None:
                return left is right
            if type(left) is not type(right):
                return False
            a = dict(left.named_parameters()) | dict(left.named_buffers())
            b = dict(right.named_parameters()) | dict(right.named_buffers())
            return a.keys() == b.keys() and all(
                a[k].shape == b[k].shape and a[k].dtype == b[k].dtype
                and (a[k] is b[k] or torch.equal(a[k], b[k])) for k in a)
        self.reuse_backbone = self.appearance_features_only and equal(
            self.appearance.backbone, self.geometry.backbone) and equal(
            self.appearance.cam_enc, self.geometry.cam_enc)
        if self.reuse_backbone:
            # Share the frozen modules before moving to CUDA as well, so the
            # losslessly aliased checkpoint does not become two GPU copies.
            self.geometry.backbone = self.appearance.backbone
            self.geometry.cam_enc = self.appearance.cam_enc
        self.shared_prefix_blocks = 0
        if self.appearance_features_only and not self.reuse_backbone and equal(self.appearance.cam_enc, self.geometry.cam_enc):
            left = getattr(self.appearance.backbone, 'pretrained', None)
            right = getattr(self.geometry.backbone, 'pretrained', None)
            if left is not None and right is not None and type(left) is type(right):
                # Input tokenization, learned camera tokens, positions and RoPE
                # buffers must also match before any block result can be reused.
                def inputs(net):
                    return {k:v for k,v in dict(net.named_parameters()).items()
                            if not k.startswith(('blocks.', 'norm.'))} | dict(net.named_buffers())
                a, b = inputs(left), inputs(right)
                if a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) and len(left.blocks) == len(right.blocks):
                    for i, (one, two) in enumerate(zip(left.blocks, right.blocks)):
                        if not equal(one, two):
                            break
                        self.shared_prefix_blocks = i + 1
                        right.blocks[i] = one
                    if self.shared_prefix_blocks:
                        self.geometry.cam_enc = self.appearance.cam_enc

    @torch.inference_mode()
    def predict_cameras(self, images):
        """Run only the released DA3 geometry branch on the prepared RGB tensor."""
        mean = images.new_tensor([.485,.456,.406])[None,None,:,None,None]
        std = images.new_tensor([.229,.224,.225])[None,None,:,None,None]
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
            feats, _ = self.geometry.backbone((images.float()-mean)/std,
                cam_token=None, export_feat_layers=[], ref_view_strategy='saddle_balanced')
        # The camera decoder consumes backbone camera tokens, not dense depth
        # or rays. Keep the upstream FP32 camera path and omit unused heads.
        from addict import Dict
        with torch.autocast(device_type=images.device.type, enabled=False):
            geometry = self.geometry._process_camera_estimation(
                feats, *images.shape[-2:], Dict())
        ext = geometry.extrinsics.float()
        if ext.shape[-2:] == (3,4):
            row = ext.new_tensor([0,0,0,1]).expand(*ext.shape[:-2],1,4)
            ext = torch.cat([ext,row],-2)
        return torch.linalg.inv(ext), geometry.intrinsics.float()

    def _da3(self, images, c2w, K, *, panorama_shared_center=False):
        n = images.shape[1]
        w2c = aligned = baseline = None
        if c2w is not None:
            # Match the native first-camera / lower median normalization.
            with torch.autocast(device_type=images.device.type, enabled=False):
                relative_w2c = torch.linalg.inv(c2w.float()) @ c2w[:,:1].float()
                relative = torch.linalg.inv(relative_w2c)
                distance = relative[..., :3,3].norm(dim=-1).median(dim=1).values.clamp_min(.1)
                normalized_w2c = relative_w2c.clone()
                normalized_w2c[..., :3,3] /= distance[:,None,None]
                normalized = torch.linalg.inv(normalized_w2c)
                w2c, aligned, baseline2 = normalize_all_camera_poses(normalized, n)
                baseline = distance * baseline2
        mean = images.new_tensor([.485,.456,.406])[None,None,:,None,None]
        std = images.new_tensor([.229,.224,.225])[None,None,:,None,None]
        normalized_rgb = (images.float()-mean)/std
        layers = [19,27,33,39]
        prefix_cache = {} if getattr(self, 'shared_prefix_blocks', 0) else None
        # The frozen native adapter uses BF16 backbone and FP32 depth/camera heads.
        if self.appearance_features_only:
            with torch.autocast(device_type=images.device.type, enabled=False):
                cam_token = None if w2c is None else self.appearance.cam_enc(w2c, K, images.shape[-2:])
            feats, aux = self.appearance.backbone(normalized_rgb, cam_token=cam_token,
                export_feat_layers=layers, ref_view_strategy='saddle_balanced',
                **({'prefix_cache': prefix_cache, 'prefix_blocks': self.shared_prefix_blocks}
                   if prefix_cache is not None else {}))
            appearance_aux = self.appearance._extract_auxiliary_features(aux, layers, *images.shape[-2:])
        else:
            appearance = self.appearance(normalized_rgb, w2c, K, export_feat_layers=layers,
                                        infer_gs=False, use_ray_pose=False, ref_view_strategy='saddle_balanced')
            appearance_aux = appearance.aux
        if prefix_cache is not None:
            feats, _ = self.geometry.backbone(normalized_rgb, cam_token=cam_token,
                export_feat_layers=[], ref_view_strategy='saddle_balanced',
                prefix_cache=prefix_cache, prefix_resume=True)
        if getattr(self, 'reuse_backbone', False) or prefix_cache is not None:
            with torch.autocast(device_type=images.device.type, enabled=False):
                geometry = self.geometry._process_depth_head(feats, *images.shape[-2:])
                geometry = self.geometry._process_camera_estimation(feats, *images.shape[-2:], geometry)
            geometry = self.geometry._process_mono_sky_estimation(geometry)
        else:
            geometry = self.geometry(normalized_rgb, w2c, K, export_feat_layers=[],
                                    infer_gs=False, use_ray_pose=False, ref_view_strategy='saddle_balanced')
        tokens = []
        for projection, layer in zip(self.projections,layers):
            feat = appearance_aux[f'feat_layer_{layer}']
            b,s,h,w,c=feat.shape
            tokens.append(projection(feat.reshape(b,s,h*w,c).float()))
        extrinsics = geometry.extrinsics.float()
        if extrinsics.shape[-2:] == (3,4):
            row = extrinsics.new_tensor([0,0,0,1]).expand(*extrinsics.shape[:-2],1,4)
            extrinsics=torch.cat([extrinsics,row],-2)
        predicted_c2w = torch.linalg.inv(extrinsics)
        predicted_K = geometry.intrinsics.float()
        depth, predicted_c2w, radius, _ = normalize_predicted_point_radius(
            geometry.depth.float(), predicted_c2w, predicted_K)
        if c2w is not None:
            meta = dict(alignment_pred_extrinsics=extrinsics, alignment_gt_extrinsics=w2c,
                        alignment_gt_c2w=aligned, alignment_gt_depth_scale=baseline)
            if panorama_shared_center:
                # Pure rotation has no identifiable baseline scale. Preserve the
                # native depth/radius normalization and exact local rotations.
                effective_c2w = aligned.float().clone()
                effective_c2w[..., :3, 3] = 0
            else:
                effective_c2w, _ = align_rdnorm_gt_via_legacy_da3(meta, radius)
            effective_K = K
        else:
            effective_c2w, effective_K = predicted_c2w, predicted_K
        return tokens, 0, depth, effective_c2w, effective_K, radius

    @torch.inference_mode()
    def forward(self, images, c2w=None, intrinsics=None, *, panorama_shared_center=False,
                feature_projector=None):
        if images.ndim != 5 or images.shape[0]!=1 or images.shape[2]!=3:
            raise ValueError('Expected RGB tensor [1,views,3,height,width]')
        if (c2w is None)!=(intrinsics is None):
            raise ValueError('Provide both c2w and pixel intrinsics, or neither')
        if panorama_shared_center:
            if c2w is None:
                raise ValueError('Panorama reconstruction requires calibrated cameras')
            centers = c2w.float()[..., :3, 3]
            if not torch.allclose(centers, centers[:, :1].expand_as(centers), atol=1e-7, rtol=0):
                raise ValueError('Panorama reconstruction requires a shared camera center')
        with torch.autocast(device_type=images.device.type,dtype=torch.bfloat16,enabled=images.is_cuda):
            tokens,start,depth,poses,K,radius=self._da3(
                images,c2w,intrinsics,panorama_shared_center=panorama_shared_center)
            # Avoid retaining dense 128-channel features for every view.
            parts=[]
            projected=[] if feature_projector is not None else None
            for i in range(images.shape[1]):
                features=self.gs_feature_head.forward_chunk(tokens,images,start,i,i+1)
                if projected is not None:
                    projected.append(feature_projector(features))
                parts.append(self.gaussian_head(features,images[:,i:i+1],depth[:,i:i+1].clamp_min(1e-6),poses[:,i:i+1],K[:,i:i+1]))
            raw={key:torch.cat([p[key] for p in parts],0) for key in parts[0]}
        # Preserve the native batched FP32 unprojection kernel, including its
        # floating-point rounding at discontinuous voxel boundaries.
        from .geometry import depth_to_world_coords_points
        with torch.autocast(device_type=images.device.type,enabled=False):
            points,_,_=depth_to_world_coords_points(depth[0].float(),poses[0].float(),K[0].float())
            raw['means']=points.reshape(-1,3)
        extra = {}
        if projected is None:
            splats,stats=merge(raw)
        else:
            splats,stats,membership=merge(raw, return_membership=True)
            features = torch.cat(projected, 0)[membership['keep']]
            counts = membership['counts']
            if len(features) != len(membership['inverse']):
                raise RuntimeError('Feature rows do not match native merge membership')
            merged = features.new_zeros(len(counts), 8).index_add_(0, membership['inverse'], features)
            extra['refine_features'] = (merged / counts[:, None])[membership['visible']]
            if len(extra['refine_features']) != len(splats['means']):
                raise RuntimeError('Feature rows do not match visible merged Gaussians')
        return dict(splats=splats,raw_points=raw['means'],raw_rgb=raw['base_rgb'],
                    c2w=poses[0].float(),intrinsics=K[0].float(),stats=stats,
                    radius=radius,depth=depth,**extra)
