#!/usr/bin/env python3
"""Convert existing calibration to prior_camera.json, or validate it before inference."""
import argparse
from pathlib import Path
import sys

# Allow direct invocation from any working directory without installing the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.camera_conversion import cameras_from_arrays, cameras_from_colmap
from src.utils.camera_io import (
    list_image_files, load_cameras_json, validate_cameras_for_images, write_cameras_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    colmap = commands.add_parser('colmap', help='Convert an undistorted COLMAP text or binary model.')
    colmap.add_argument('--model', required=True, help='Sparse model folder containing cameras/images files.')
    colmap.add_argument('--model_format', choices=['auto', 'bin', 'txt'], default='auto')
    arrays = commands.add_parser('arrays', help='Convert calibrated pose/K .npy arrays with an explicit image order.')
    arrays.add_argument('--poses', required=True, help='Numeric [N,4,4] or [N,3,4] .npy file.')
    arrays.add_argument('--intrinsics', required=True, help='Pixel K: [3,3] shared, or [N,3,3] .npy file.')
    arrays.add_argument('--image_names', required=True, help='UTF-8 text: one image filename per pose row.')
    arrays.add_argument('--pose_convention', choices=['c2w', 'w2c'], required=True)
    arrays.add_argument('--camera_axes', choices=['opencv', 'opengl'], required=True)
    arrays.add_argument('--intrinsics_pixel_center', choices=['opencv', 'colmap'], default='opencv',
                        help='OpenCV: first pixel center is (0,0); COLMAP: (0.5,0.5).')
    check = commands.add_parser('check', help='Validate an existing JSON and its image coverage.')
    check.add_argument('--cameras', required=True)
    for command in (colmap, arrays, check):
        command.add_argument('--images', required=True, help='Flat folder of original-resolution pinhole images.')
    for command in (colmap, arrays):
        command.add_argument('--output', help='Default: prior_camera.json beside the images directory.')
        command.add_argument('--overwrite', action='store_true', help='Replace an existing output JSON.')
    args = parser.parse_args(argv)
    try:
        images = list_image_files(args.images)
        if args.command == 'check':
            poses, intrinsics = load_cameras_json(args.cameras)
            validate_cameras_for_images(images, poses, intrinsics)
            print(f'Valid: {len(images)} images have OpenCV c2w poses and pixel intrinsics.')
            print('Pose accuracy and the declared coordinate convention require your calibration records.')
            return
        if args.command == 'colmap':
            poses, intrinsics = cameras_from_colmap(images, args.model, args.model_format)
        else:
            poses, intrinsics = cameras_from_arrays(
                images, args.poses, args.intrinsics, args.image_names,
                args.pose_convention, args.camera_axes, args.intrinsics_pixel_center)
        output = Path(args.output) if args.output else Path(args.images).resolve().parent / 'prior_camera.json'
        write_cameras_json(output, images, poses, intrinsics, overwrite=args.overwrite)
        print(f'Saved {len(images)} cameras to {output}')
        print('Output: OpenCV camera-to-world matrices; original-image pixel intrinsics.')
        print('Next: run the pose inference script with the same --images folder as its input directory.')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
