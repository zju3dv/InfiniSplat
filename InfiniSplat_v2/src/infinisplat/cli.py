"""InfiniSplat v2 image-to-Gaussian inference."""
import argparse
from pathlib import Path
from .pipeline import Pipeline
from .export import export_ply
from .cameras import add_camera_arguments, resolve_camera_arguments

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--refiner', help='Optional separate refiner bundle directory or Hub repository')
    parser.add_argument('--refinement-steps', type=int, choices=range(4), metavar='0..3',
                        help='Default: 3 with --refiner, otherwise 0; 0 keeps one-shot inference')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--images', nargs='+', help='Ordered input image paths')
    inputs.add_argument('--images-dir', type=Path, help='Read JPG, PNG and WebP files in filename order (nonrecursive)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--long-side', type=int, default=518)
    parser.add_argument('--max-views', type=int, default=32, choices=range(2, 51), metavar='2..50')
    parser.add_argument('--no-html', action='store_true', help='Export PLY only; skip the standalone SH0 viewer')
    add_camera_arguments(parser)
    args = parser.parse_args()
    if args.refinement_steps and not args.refiner:
        parser.error('--refinement-steps requires --refiner')
    if not 112 <= args.long_side <= 2044:
        parser.error('--long-side must be between 112 and 2044')
    if args.images_dir is not None:
        if not args.images_dir.is_dir():
            parser.error('--images-dir must be an existing directory')
        args.images = sorted((p for p in args.images_dir.iterdir()
                              if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}),
                             key=lambda p: p.name)
    if not 2 <= len(args.images) <= args.max_views:
        parser.error(f'Select 2–{args.max_views} overlapping images; use --images to select a subset')
    mode, cameras = resolve_camera_arguments(args, parser)
    if Path(args.output).exists():
        parser.error('--output must be a new directory')
    if not args.no_html:
        from .viewer import require_html_converter
        try:
            require_html_converter()
        except RuntimeError as error:
            parser.error(str(error))
    try:
        pipeline = Pipeline.from_pretrained(args.model, args.device, refiner=args.refiner)
        scene = pipeline.reconstruct(args.images, cameras=cameras, pose_mode=mode,
                                     long_side=args.long_side, max_views=args.max_views,
                                     refinement_steps=args.refinement_steps)
        ply = export_ply(scene, args.output, html=not args.no_html)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(ply)
    if not args.no_html:
        print(Path(args.output) / 'scene.html')


if __name__ == '__main__':
    main()
