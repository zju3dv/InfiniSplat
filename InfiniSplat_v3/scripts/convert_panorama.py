#!/usr/bin/env python3
"""Convert one 2:1 panorama into 24 perspective images and matching camera priors."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.panorama import convert_panorama


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Full 360 x 180 degree equirectangular image.')
    parser.add_argument('--output', required=True, help='Scene directory receiving images/ and prior_camera.json.')
    parser.add_argument('--size', type=int, default=1022, help='Square view width/height; default 1022.')
    parser.add_argument('--fov', type=float, default=60, help='Horizontal and vertical view FOV in degrees; default 60.')
    parser.add_argument('--overwrite', action='store_true', help='Replace generated views and metadata; preserve other files.')
    args = parser.parse_args(argv)
    try:
        output = convert_panorama(args.input, args.output, args.size, args.fov, args.overwrite)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Prepared 24 calibrated perspective views in {output}')


if __name__ == '__main__':
    main()
