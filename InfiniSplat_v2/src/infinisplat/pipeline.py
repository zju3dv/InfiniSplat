"""Small public API: model directory + RGB images -> Gaussian scene."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from safetensors.torch import load_file
from .model import InfiniSplat
from .weights import expand_aliases
from .cameras import validate_cameras, validate_intrinsics


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_images(paths, long_side=518, intrinsics=None, max_views=12):
    paths = [Path(p) for p in paths]
    if not isinstance(max_views, int) or not 2 <= max_views <= 50:
        raise ValueError('max_views must be an integer from 2 to 50.')
    if not 2 <= len(paths) <= max_views:
        raise ValueError(f'Upload 2–{max_views} overlapping images of one static scene.')
    if long_side < 112 or long_side > 2044:
        raise ValueError('long_side must be between 112 and 2044')
    matrices=None if intrinsics is None else validate_intrinsics(intrinsics,len(paths)).copy()
    images, records=[],[]
    with Image.open(paths[0]) as im:
        w,h=ImageOps.exif_transpose(im).size
    factor=long_side/max(w,h)
    w,h=[max(14,round(v*factor/14)*14) for v in (w,h)]
    for i,p in enumerate(paths):
        if p.stat().st_size>20*1024**2:
            raise ValueError('Each image must be at most 20 MB.')
        with Image.open(p) as im:
            if matrices is not None and im.getexif().get(274, 1) != 1:
                raise ValueError('Provided cameras require images without EXIF rotation; rotate images and update calibration first')
            if im.width*im.height>40_000_000:
                raise ValueError('Each image must be at most 40 megapixels.')
            if im.format not in ('JPEG','PNG','WEBP'):
                raise ValueError('Use JPG, PNG or WebP images.')
            im=ImageOps.exif_transpose(im).convert('RGB')
            original=list(im.size)
            scale=max(w/im.width,h/im.height)
            # Roundoff can make an uncropped edge slightly negative. Pillow
            # requires every crop coordinate to remain inside the source image.
            left,top=max(0.,(im.width-w/scale)/2),max(0.,(im.height-h/scale)/2)
            box=(left,top,min(float(im.width),left+w/scale),
                 min(float(im.height),top+h/scale))
            resized=im.resize((w,h),Image.Resampling.BICUBIC,box=box)
            images.append(torch.from_numpy(np.array(resized)).permute(2,0,1).float()/255)
        if matrices is not None:
            matrices[i]=np.array([[scale,0,-left*scale],[0,scale,-top*scale],[0,0,1]])@matrices[i]
        records.append(dict(sha256=sha256(p),original_wh=original,processed_wh=[w,h],crop_box=box))
    return torch.stack(images)[None],matrices,records


class Pipeline:
    def __call__(self, images, **kwargs):
        return self.reconstruct(images, **kwargs)

    def __init__(self, model, config, device):
        if isinstance(model, InfiniSplat):
            model.configure_inference()
        self.model=model.eval().requires_grad_(False).to(device)
        self.config=config
        self.device=torch.device(device)

    @classmethod
    def from_pretrained(cls, directory, device='cuda:0'):
        path=Path(directory)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path=Path(snapshot_download(str(directory),allow_patterns=['config.json','model.safetensors']))
        cfg=json.loads((path/'config.json').read_text())
        if cfg.get('backend') != 'da3':
            raise ValueError('InfiniSplat requires DA3 inference weights')
        contract={'normalization':'mean_predicted_point_radius','merge':'uniform_mean',
                  'voxel_multiplier':1.4,'opacity_threshold':.01,'eps2d':.01}
        if cfg.get('format_version') not in (1, 2) or any(cfg.get(k)!=v for k,v in contract.items()):
            raise ValueError('Unsupported InfiniSplat inference configuration')
        if sha256(path/'model.safetensors')!=cfg['weights_sha256']:
            raise ValueError('Weight integrity check failed')
        if not str(device).startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('This release requires an NVIDIA CUDA GPU.')
        # Construct on CPU: RoPE and image normalization buffers remain materialized.
        model=InfiniSplat(cfg.get('backbone_config'),
                     appearance_features_only=cfg['format_version']==2)
        state=load_file(str(path/'model.safetensors'))
        state=expand_aliases(state, cfg.get('tensor_aliases', {}))
        model.load_state_dict(state,strict=True,assign=True)
        return cls(model,cfg,device)

    @torch.inference_mode()
    def reconstruct(self, images, cameras=None, long_side=518, max_views=12, pose_mode=None,
                    panorama_shared_center=False):
        images = list(images)
        pose_mode = pose_mode or ('provided' if cameras is not None else 'da3-pose')
        if pose_mode not in ('provided', 'da3-pose', 'pose-free'):
            raise ValueError('pose_mode must be provided, da3-pose or pose-free')
        if (pose_mode == 'provided') != (cameras is not None):
            raise ValueError('Only provided mode accepts and requires cameras')
        c2w=K=None
        if cameras is not None:
            cameras=validate_cameras(cameras, images)
            c2w,K=cameras['c2w'],cameras['intrinsics']
        if panorama_shared_center:
            if cameras is None:
                raise ValueError('Panorama mode requires provided cameras')
            if not np.allclose(c2w[:, :3, 3], c2w[:1, :3, 3], atol=1e-7, rtol=0):
                raise ValueError('Panorama mode requires a shared camera center')
        pixels,K,records=prepare_images(images,long_side,K,max_views=max_views)
        pixels_device = pixels.to(self.device)
        pose_tensor = None if c2w is None else torch.from_numpy(c2w)[None].to(self.device)
        K_tensor = None if K is None else torch.from_numpy(K)[None].to(self.device)
        prior = None
        if pose_mode == 'da3-pose':
            pose_tensor, K_tensor = self.model.predict_cameras(pixels_device)
            if (pose_tensor.shape != (1,len(images),4,4) or K_tensor.shape != (1,len(images),3,3)
                    or not torch.isfinite(pose_tensor).all() or not torch.isfinite(K_tensor).all()
                    or (K_tensor[...,0,0] <= 0).any() or (K_tensor[...,1,1] <= 0).any()):
                raise ValueError('Invalid DA3 camera prior')
            prior = dict(c2w=pose_tensor[0].detach().cpu().clone(),
                         intrinsics=K_tensor[0].detach().cpu().clone())
        kwargs = {'panorama_shared_center': True} if panorama_shared_center else {}
        result=self.model(pixels_device, pose_tensor, K_tensor, **kwargs)
        result['images']=pixels[0]
        result['metadata']=dict(backend=self.config['backend'],
             weights_sha256=self.config['weights_sha256'],
             input_mode={'pose-free':'rgb_only','provided':'rgb_pose_k','da3-pose':'prepass_pose_conditioned'}[pose_mode],
             pose_mode=pose_mode,
             inputs=records,space='rdnorm_first_camera_relative',stats=result['stats'],
             record_class='diagnostic',max_gaussians=0,
             note='Interactive reconstruction metadata.')
        if panorama_shared_center:
            result['metadata']['panorama'] = dict(
                scale_status='non_identifiable', translation_scale=None,
                baseline_scale_fit_attempted=False, pose_prepass_attempted=False,
                normalization=self.config['normalization'],
                condition_pose_source='provided_shared_center',
                condition_intrinsics_source='provided',
                unprojection_pose_source='provided_first_camera_relative_zero_center',
                unprojection_intrinsics_source='provided',
                render_pose_source='effective_unprojection_cameras')
        if prior is not None:
            result['camera_prior'] = prior
            result['metadata']['camera_prior'] = dict(
                file='camera_prior.npz', convention='OpenCV c2w',
                space='da3_raw_predicted', intrinsics_space='processed_image_pixels',
                producer='released_da3_geometry_branch', weights_sha256=self.config['weights_sha256'],
                contract='prepass_pose_condition_and_unproject',
                geometry='prior trajectory aligned into native RDNorm space',
                same_prepared_images=True, same_effective_intrinsics=True)
        return result
