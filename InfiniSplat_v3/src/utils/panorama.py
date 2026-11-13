"""Equirectangular panoramas to calibrated pinhole views, using NumPy and Pillow."""
import json
import math
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
from PIL import Image

from .camera_io import write_cameras_json


def view_cameras(size=1022, fov=60.0):
    """Match the bundled panorama layout: equator, downward ring, upward ring."""
    if not isinstance(size, int) or size < 2:
        raise ValueError('View size must be an integer of at least 2 pixels.')
    if not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError('Field of view must be finite and between 0 and 180 degrees.')
    focal = size / (2 * math.tan(math.radians(fov) / 2))
    intrinsic = np.array([[focal, 0, (size - 1) / 2],
                          [0, focal, (size - 1) / 2], [0, 0, 1]], dtype=np.float64)
    result = []
    for pitch in (0, -45, 45):
        p = math.radians(pitch)
        rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)],
                       [0, math.sin(p), math.cos(p)]])
        for yaw in range(0, 360, 45):
            y = math.radians(yaw)
            ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0],
                           [-math.sin(y), 0, math.cos(y)]])
            pose = np.eye(4)
            pose[:3, :3] = ry @ rx
            result.append((f'frame_{len(result):04d}', yaw, pitch, pose, intrinsic.copy()))
    return result


def perspective_view(panorama, pose, intrinsic, size, chunk_rows=64):
    """Bilinear sampling; wrap longitude and clamp latitude at the poles."""
    height, width, _ = panorama.shape
    output = np.empty((size, size, 3), dtype=np.uint8)
    x = (np.arange(size, dtype=np.float64) - intrinsic[0, 2]) / intrinsic[0, 0]
    for start in range(0, size, chunk_rows):
        stop = min(start + chunk_rows, size)
        y = (np.arange(start, stop, dtype=np.float64) - intrinsic[1, 2]) / intrinsic[1, 1]
        xx, yy = np.meshgrid(x, y)
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ pose[:3, :3].T
        longitude = np.arctan2(rays[..., 0], rays[..., 2])
        latitude = np.arctan2(-rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
        u = (longitude / (2 * np.pi) + .5) * width - .5
        v = np.clip((.5 - latitude / np.pi) * height - .5, 0, height - 1)
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        wx, wy = (u - x0)[..., None], (v - y0)[..., None]
        top = panorama[y0, x0 % width] * (1 - wx) + panorama[y0, (x0 + 1) % width] * wx
        bottom_y = np.minimum(y0 + 1, height - 1)
        bottom = panorama[bottom_y, x0 % width] * (1 - wx) + panorama[bottom_y, (x0 + 1) % width] * wx
        output[start:stop] = np.clip(np.rint(top * (1 - wy) + bottom * wy), 0, 255).astype(np.uint8)
    return output


def convert_panorama(source, output, size=1022, fov=60.0, overwrite=False):
    """Stage a complete scene before replacing only the three managed outputs."""
    cameras = view_cameras(size, fov)
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    names = ('images', 'prior_camera.json', 'panorama.json')
    if source == output / 'images' or output / 'images' in source.parents:
        raise ValueError('Keep the source panorama outside the managed output images/ directory.')
    if not source.is_file():
        raise FileNotFoundError(source)
    existing = [output / name for name in names if (output / name).exists() or (output / name).is_symlink()]
    if existing and not overwrite:
        raise FileExistsError(f'Output already exists: {existing[0]}. Use --overwrite to replace generated files.')
    with warnings.catch_warnings():
        # 16K ERP images exceed Pillow's warning threshold but remain below its error limit.
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        with Image.open(source) as image:
            if image.width != 2 * image.height:
                raise ValueError('Expected a full 360 x 180 degree equirectangular image with a 2:1 aspect ratio.')
            if image.getexif().get(274, 1) != 1:
                raise ValueError('Apply EXIF orientation to the panorama before conversion.')
            panorama = np.asarray(image.convert('RGB'))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pano-', dir=output) as temporary:
        stage = Path(temporary)
        (stage / 'images').mkdir()
        poses, intrinsics, views = {}, {}, []
        for camera_id, yaw, pitch, pose, intrinsic in cameras:
            image_path = stage / 'images' / f'{camera_id}.png'
            Image.fromarray(perspective_view(panorama, pose, intrinsic, size)).save(image_path)
            poses[camera_id], intrinsics[camera_id] = pose, intrinsic
            views.append(dict(camera_id=camera_id, image=f'images/{camera_id}.png',
                              yaw_degrees=yaw, pitch_degrees=pitch))
        write_cameras_json(stage / 'prior_camera.json', sorted((stage / 'images').glob('*.png')), poses, intrinsics)
        metadata = dict(source=source.name, source_size=[panorama.shape[1], panorama.shape[0]],
                        view_size=[size, size], fov_degrees=fov,
                        camera_convention='OpenCV camera-to-world: +X right, +Y down, +Z forward',
                        panorama_convention='Center looks along +Z; positive yaw turns right; positive pitch looks up',
                        views=views)
        (stage / 'panorama.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Keep originals and unrelated files. Restore previous outputs if installation fails.
        saved, installed = [], []
        try:
            for name in names:
                target = output / name
                if target.exists() or target.is_symlink():
                    target.rename(stage / f'old-{name}')
                    saved.append(name)
                (stage / name).rename(target)
                installed.append(name)
        except BaseException:
            for name in reversed(installed):
                target = output / name
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            for name in saved:
                (stage / f'old-{name}').rename(output / name)
            raise
    return output
