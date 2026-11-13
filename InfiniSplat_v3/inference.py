#!/usr/bin/env python3
"""SplatAnything: single-scene feed-forward Gaussian inference."""
import argparse
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_scene_inputs(input_path, pose_condition):
    """Resolve the fixed scene layout; RGB mode never opens camera metadata."""
    scene = Path(input_path).expanduser().resolve()
    if not scene.is_dir():
        raise FileNotFoundError(f'Scene directory not found: {scene}')
    images = scene / 'images'
    if not images.is_dir():
        raise FileNotFoundError(f'Scene must contain an images/ directory: {images}. '
                                'Point --input_path at the scene directory.')
    camera = scene / 'prior_camera.json' if pose_condition else None
    if camera is not None and not camera.is_file():
        raise FileNotFoundError(f'Pose-conditioned mode requires {camera}, containing poses '
                                'and intrinsics for every input image.')
    return str(images), str(camera) if camera is not None else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', default=str(ROOT / 'configs/inference.yaml'))
    parser.add_argument('--input_path', required=True, help='Scene directory containing images/ and, for pose mode, prior_camera.json.')
    parser.add_argument('--output_path', default=str(ROOT / 'outputs'), help='Output root; results go into OUTPUT/SCENE/.')
    parser.add_argument('--resolution', type=int, nargs=2, default=[756, 756], metavar=('W', 'H'),
                        help='Positive multiples of 14; default 756 756.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--mode', choices=['rgb', 'pose'], help='RGB-only, or pose-conditioned using SCENE/prior_camera.json.')
    mode.add_argument('--pose_condition', dest='pose_condition', action='store_true',
                      help=argparse.SUPPRESS)
    mode.add_argument('--no_pose_condition', dest='pose_condition', action='store_false',
                      help=argparse.SUPPRESS)
    parser.set_defaults(pose_condition=None)
    parser.add_argument('--sky_seg', action='store_true', help='Segment and remove sky before voxel merging.')
    parser.add_argument('--sky_onnx_path', default=str(ROOT / 'checkpoints/skyseg.onnx'))
    parser.add_argument('--sky_max_fraction', type=float, default=0.95)
    parser.add_argument('--ckpt_path', help='Complete Gaussian student checkpoint (.ckpt/.pth/.safetensors).')
    parser.add_argument('--geometry_ckpt_path', help='WorldMirror-2 geometry checkpoint.')
    parser.add_argument('--opacity_thresh', type=float, help='Override the exported Gaussian opacity threshold.')
    parser.add_argument('--no_cam_align', action='store_true', help='Keep supplied camera frame; default anchors to camera 0.')
    parser.add_argument('--no_depth_rgb', action='store_true', help='Skip per-view RGB and depth export.')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    if args.mode is not None:
        args.pose_condition = args.mode == 'pose'
    if any(n <= 0 or n % 14 for n in args.resolution):
        parser.error('--resolution W H must be positive multiples of 14 (e.g. 756 756 or 1022 756).')
    if not 0 <= args.sky_max_fraction <= 1:
        parser.error('--sky_max_fraction must be in [0, 1].')
    if args.opacity_thresh is not None and not 0 <= args.opacity_thresh <= 1:
        parser.error('--opacity_thresh must be in [0, 1].')
    return args


def load_config(args):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config_path)
    if not OmegaConf.is_dict(cfg) or set(cfg) != {'wrapper'}:
        raise ValueError('The config must contain only the wrapper mapping.')
    fields = {
        'wrapper': {'worldmirror2_pretrained', 'SplatAnything_pretrained',
                    'enable_cond', 'infer_opacity_thresh', 'model'},
        'wrapper.model': {'use_gt_camera', 'filter_flying_points_infer',
                          'filter_large_scale_infer',
                          'max_scale_abs', 'max_scale_pct', 'gs_voxel_size'},
    }
    for path, expected in fields.items():
        section = OmegaConf.select(cfg, path)
        if not OmegaConf.is_dict(section):
            raise ValueError(f'{path} must be a mapping.')
        missing = expected - set(section)
        extra = set(section) - expected
        if missing or extra:
            raise ValueError(f'Invalid config fields in {path}: '
                             f'missing={sorted(missing)}, unsupported={sorted(extra)}.')
    if args.pose_condition is not None:
        cfg.wrapper.enable_cond = args.pose_condition
        cfg.wrapper.model.use_gt_camera = args.pose_condition
    if bool(cfg.wrapper.enable_cond) != bool(cfg.wrapper.model.use_gt_camera):
        raise ValueError('enable_cond and use_gt_camera must both be true (pose-conditioned) '
                         'or both false (RGB-only).')
    args.images_path, args.cameras = resolve_scene_inputs(args.input_path, bool(cfg.wrapper.enable_cond))
    if args.ckpt_path:
        cfg.wrapper.SplatAnything_pretrained = args.ckpt_path
    if args.geometry_ckpt_path:
        cfg.wrapper.worldmirror2_pretrained = args.geometry_ckpt_path
    if args.opacity_thresh is not None:
        cfg.wrapper.infer_opacity_thresh = args.opacity_thresh
    if not 0 <= cfg.wrapper.infer_opacity_thresh <= 1:
        raise ValueError('infer_opacity_thresh must be in [0, 1].')
    if cfg.wrapper.model.gs_voxel_size <= 0:
        raise ValueError('gs_voxel_size must be positive.')
    for name in ('filter_flying_points_infer', 'filter_large_scale_infer'):
        if not isinstance(cfg.wrapper.model[name], bool):
            raise ValueError(f'{name} must be true or false.')
    max_scale_abs = cfg.wrapper.model.max_scale_abs
    if max_scale_abs is not None and max_scale_abs <= 0:
        raise ValueError('max_scale_abs must be null or positive.')
    pct = cfg.wrapper.model.max_scale_pct
    if pct is not None and not 0 < pct <= 100:
        raise ValueError('max_scale_pct must be null or in (0, 100].')
    # Paths in YAML are relative to the repository root; explicit CLI paths are relative to cwd.
    for key, override in [('SplatAnything_pretrained', args.ckpt_path),
                          ('worldmirror2_pretrained', args.geometry_ckpt_path)]:
        path = Path(cfg.wrapper[key]).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if override else ROOT) / path
        if not path.is_file():
            raise FileNotFoundError(f'{key} checkpoint not found: {path}. See README.md for weight setup.')
        cfg.wrapper[key] = str(path.resolve())
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args)
    import torch
    from omegaconf import OmegaConf
    from src.models.models.gaussian_worldmirror import GaussianWorldMirror
    from src.models.models.geometry_worldmirror import GeometryWorldMirror
    from src.utils.checkpoints import load_model_weights
    from src.utils.scene_io import build_views, export_gaussians, export_points, export_depth_rgb
    from src.utils.sky_segmentation import SkySegmenter

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use a working CUDA/PyTorch environment, or --device cpu for a slow smoke run.')
    segmenter = SkySegmenter(args.sky_onnx_path, device=args.device) if args.sky_seg else None
    views, scene_name, img_paths = build_views(
        args.images_path, args.cameras, tuple(args.resolution), device,
        cam_align=not args.no_cam_align,
        sky_segmenter=segmenter, sky_max_fraction=args.sky_max_fraction)
    cond_flags = [1, 0, 1] if cfg.wrapper.enable_cond else [0, 0, 0]
    # Architecture, positional encoding, spacing normalization, and chunking use code defaults.
    model = GaussianWorldMirror(**OmegaConf.to_container(cfg.wrapper.model),
                                enable_cond=bool(cfg.wrapper.enable_cond),
                                filter_sky_infer=args.sky_seg)
    teacher = GeometryWorldMirror(enable_cond=bool(cfg.wrapper.enable_cond))
    load_model_weights(model, cfg.wrapper.SplatAnything_pretrained)
    load_model_weights(teacher, cfg.wrapper.worldmirror2_pretrained)
    model.to(device).eval().requires_grad_(False)
    teacher.to(device).eval().requires_grad_(False)
    print(f'[Input] {scene_name}: {views["img"].shape[1]} images, resolution={args.resolution}, '
          f'enable_cond={cfg.wrapper.enable_cond}, use_gt_camera={cfg.wrapper.model.use_gt_camera}, sky_seg={args.sky_seg}')
    started = time.perf_counter()
    with torch.inference_mode():
        geometry = teacher(views, cond_flags=cond_flags)
        preds = model(views, geometry, cond_flags=cond_flags)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f'[Inference] {time.perf_counter() - started:.2f}s')
    outdir = Path(args.output_path) / scene_name
    outdir.mkdir(parents=True, exist_ok=True)
    # An empty filtered scene must not be mistaken for a successful reconstruction.
    if not len(preds['splats']['means'][0]):
        raise RuntimeError('All Gaussians were filtered. Check the input, sky masks, and scale threshold.')
    if not (preds['splats']['opacities'][0] >= cfg.wrapper.infer_opacity_thresh).any():
        raise RuntimeError('All Gaussians are below opacity_thresh. Lower the threshold and check the checkpoint.')
    export_gaussians(preds, outdir, cfg.wrapper.infer_opacity_thresh, None)
    export_points(preds, outdir)
    if not args.no_depth_rgb:
        export_depth_rgb(preds, views, img_paths, outdir)
    cameras = {'extrinsics': [], 'intrinsics': []}
    for index, image in enumerate(img_paths):
        camera_id = Path(image).stem
        for name, key in [('extrinsics', 'export_camera_poses'), ('intrinsics', 'export_camera_intrs')]:
            cameras[name].append({'camera_id': camera_id, 'matrix': preds[key][0, index].cpu().tolist()})
    (outdir / 'cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')
    OmegaConf.save(cfg, outdir / 'config.yaml', resolve=True)
    (outdir / 'run.json').write_text(json.dumps({**vars(args), 'images': img_paths, 'cond_flags': cond_flags}, indent=2) + '\n')
    print(f'[Done] {outdir}')


if __name__ == '__main__':
    main()
