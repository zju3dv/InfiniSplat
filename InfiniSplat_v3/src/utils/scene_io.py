# Modified for SplatAnything: inference-only release.
"""Image preprocessing and PLY/depth export."""
from pathlib import Path
import cv2
import numpy as np
import PIL.Image
import torch
from src.utils import cropping
from src.utils.camera_io import list_image_files, load_cameras_json
from src.utils.sky_segmentation import SkySegmenter
from src.utils.save_utils import save_gs_ply, save_scene_ply, save_image_png, save_depth_png, save_depth_npy

def prepare_input(input_path):
    folder = Path(input_path)
    images = [str(p) for p in list_image_files(folder)]
    return (images, folder.name)

def _centered_default_intrinsics(image_size):
    """Dummy centered intrinsics used only to reproduce camera-free image preprocessing.

    In RGB-only mode this matrix is never passed to the model or renderer. It only lets the
    camera-aware crop helper reduce to an aspect-preserving center crop.
    """
    (width, height) = image_size
    focal = float(max(width, height))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

def preprocess_view(image: 'PIL.Image.Image', K: np.ndarray=None, resolution_wh=(756, 756), depthmap=None):
    image = image.convert('RGB')
    K = _centered_default_intrinsics(image.size) if K is None else K.astype(np.float32).copy()
    if not (0 < K[0, 2] < image.width and 0 < K[1, 2] < image.height):
        raise ValueError('Camera principal point must lie inside the original image.')
    if depthmap is not None:
        depthmap = np.ascontiguousarray(depthmap, dtype=np.float32)
        if depthmap.shape[:2] != image.size[::-1]:
            depthmap = cv2.resize(depthmap, image.size, interpolation=cv2.INTER_NEAREST)
    (W, H) = image.size
    (cx, cy) = K[:2, 2].round().astype(int)
    (mx, my) = (min(cx, W - cx), min(cy, H - cy))
    (image, depthmap, K) = cropping.crop_image_depthmap(image, depthmap, K, (cx - mx, cy - my, cx + mx, cy + my))
    res = np.array(resolution_wh)
    (image, depthmap, K) = cropping.rescale_image_depthmap(image, depthmap, K, res)
    K2 = cropping.camera_matrix_of_crop(K, image.size, resolution_wh, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(K, K2, resolution_wh)
    (image, depthmap, K2) = cropping.crop_image_depthmap(image, depthmap, K, crop_bbox)
    img_t = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
    K_t = torch.from_numpy(K2.astype(np.float32))
    if depthmap is None:
        return (img_t, K_t)
    return (img_t, K_t, torch.from_numpy(np.ascontiguousarray(depthmap, dtype=np.float32)))

def build_views(input_path: str, cameras_path: str=None, resolution_wh=(756, 756), device='cpu', cam_align: bool=True, sky_segmenter: 'SkySegmenter'=None, sky_max_fraction: float=0.95):
    (img_paths, scene_name) = prepare_input(input_path)
    if scene_name in ('images', 'image', 'imgs') and Path(input_path).parent.name:
        scene_name = Path(input_path).parent.name
    has_camera_prior = cameras_path is not None
    if has_camera_prior:
        if not Path(cameras_path).is_file():
            raise FileNotFoundError(f'Camera JSON not found: {cameras_path}')
        (extr_map, intr_map) = load_cameras_json(cameras_path)
    else:
        (extr_map, intr_map) = ({}, {})
    (imgs, Ks, c2ws, depths, sky_masks) = ([], [], [], [], [])
    for p in img_paths:
        stem = Path(p).stem
        if has_camera_prior and (stem not in extr_map or stem not in intr_map):
            raise ValueError(f"No camera entry (extrinsics/intrinsics) for image '{stem}' in {cameras_path}.")
        prior_d = None
        K = intr_map[stem] if has_camera_prior else None
        out = preprocess_view(PIL.Image.open(p), K, resolution_wh, depthmap=prior_d)
        if prior_d is None:
            (img_t, K2) = out
        else:
            (img_t, K2, depth_t) = out
            depths.append(depth_t)
        imgs.append(img_t)
        if has_camera_prior:
            Ks.append(K2)
            c2ws.append(torch.from_numpy(extr_map[stem]))
        if sky_segmenter is not None:
            rgb_hw3 = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            sky = sky_segmenter.sky_mask(rgb_hw3)
            frac = float(sky.mean())
            if frac > sky_max_fraction:
                print(f"[SkySeg] '{stem}': sky fraction {frac:.2%} > {sky_max_fraction:.0%} -> leaving view UNFILTERED (likely false positive).")
                sky = np.zeros_like(sky)
            sky_masks.append(torch.from_numpy(sky))
    imgs = torch.stack(imgs).unsqueeze(0).to(device)
    views = {'img': imgs}
    if has_camera_prior:
        intr = torch.stack(Ks).unsqueeze(0).to(device).float()
        extr = torch.stack(c2ws).unsqueeze(0).to(device).float()
        if cam_align:
            first_inv = torch.linalg.inv(extr[:, 0])
            extr = first_inv.unsqueeze(1) @ extr
        views['camera_poses'] = extr
        views['camera_intrs'] = intr
    if depths:
        views['depthmap'] = torch.stack(depths).unsqueeze(0).to(device).float()
        print(f"[Input] Loaded prior_depth for {len(depths)} views (sky pixels depth<0: {(views['depthmap'] < 0).sum().item()})")
    if sky_masks:
        sky = torch.stack(sky_masks).unsqueeze(0).to(device)
        if 'depthmap' not in views:
            views['depthmap'] = torch.zeros_like(sky, dtype=torch.float32)
        views['depthmap'] = views['depthmap'].masked_fill(sky, -1.0)
        print(f'[SkySeg] tagged {int(sky.sum().item())} sky pixels across {len(sky_masks)} views ({100.0 * sky.float().mean().item():.1f}% overall).')
    return (views, scene_name, img_paths)

def _batch0(splats, key):
    """Pull batch 0 of a splat field (list-per-batch after prune_gs, or [B, ...] before)."""
    v = splats.get(key)
    if v is None:
        return None
    return v[0]

def export_gaussians(preds, outdir: Path, opacity_thresh: float, scale_quantile):
    """Write the pruned Gaussian splats (batch 0) to gaussians.ply.

    SH DC is stored raw into f_dc (the viewer reconstructs rgb = 0.5 + C0 * f_dc), matching
    save_validation_gs_ply. Near-transparent Gaussians below opacity_thresh are dropped.
    """
    splats = preds.get('splats')
    if splats is None or 'means' not in splats:
        print('  [gaussians] no splats in predictions -> skip')
        return
    means = _batch0(splats, 'means')
    scales = _batch0(splats, 'scales')
    quats = _batch0(splats, 'quats')
    opacities = _batch0(splats, 'opacities')
    sh = _batch0(splats, 'sh')
    if means is None or means.shape[0] == 0:
        print('  [gaussians] empty cloud -> skip')
        return
    dc = sh[:, 0, :] if sh is not None else torch.zeros_like(means)
    if opacity_thresh and opacities is not None:
        keep = opacities.reshape(-1) >= opacity_thresh
        if keep.any():
            (means, scales, quats) = (means[keep], scales[keep], quats[keep])
            (dc, opacities) = (dc[keep], opacities.reshape(-1)[keep])
        else:
            print('  [gaussians] all Gaussians below opacity threshold -> skip')
            return
    path = outdir / 'gaussians.ply'
    save_gs_ply(path, means, scales, quats, dc, opacities, quantile_threshold=scale_quantile)
    print(f'  [gaussians] saved {means.shape[0]} Gaussians -> {path}')

def export_depth_rgb(preds, views, img_paths, outdir: Path):
    depth = preds.get('depth')
    imgs = views.get('img')
    if depth is None:
        print('  [depth/rgb] no predicted depth -> skip')
        return
    depth = depth[0]
    imgs = imgs[0]
    S = depth.shape[0]
    depth_dir = outdir / 'depth'
    rgb_dir = outdir / 'rgb'
    depth_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for i in range(S):
        stem = Path(img_paths[i]).stem if i < len(img_paths) else f'{i:04d}'
        d = depth[i, ..., 0].float()
        save_depth_png(depth_dir / f'{stem}.png', d)
        save_depth_npy(depth_dir / f'{stem}.npy', d)
        save_image_png(rgb_dir / f'{stem}.png', imgs[i].permute(1, 2, 0).float())
    print(f'  [depth/rgb] saved {S} predicted depth maps -> {depth_dir} and input RGB -> {rgb_dir}')

def export_points(preds, outdir):
    (means, rgb) = (preds['points_ply_means'][0], preds['points_ply_rgb'][0])
    colors = (rgb.clamp(0, 1) * 255).round().to(torch.uint8)
    save_scene_ply(outdir / 'points.ply', means, colors)
    print(f'  [points] saved {len(means)} points')
