"""Camera JSON validation shared by inference and the CPU-only conversion tool."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_image_files(image_dir):
    folder = Path(image_dir)
    if not folder.is_dir():
        raise ValueError(f'Image directory not found: {folder}')
    images = sorted(p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f'No supported images found in {folder}')
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError('Image filename stems must be unique for camera matching and output names.')
    return images


def validate_pose(matrix, camera_id):
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'Camera {camera_id}: pose must be a finite 4x4 matrix.')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)):
        raise ValueError(f'Camera {camera_id}: expected a rigid transform with a proper rotation '
                         'and last row [0, 0, 0, 1].')


def validate_intrinsics(matrix, camera_id):
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])):
        raise ValueError(f'Camera {camera_id}: intrinsics must be a finite 3x3 pixel matrix '
                         'with positive focal lengths and last row [0, 0, 1].')
    if not np.allclose([matrix[0, 1], matrix[1, 0]], 0, atol=1e-6):
        raise ValueError(f'Camera {camera_id}: skewed intrinsics are unsupported; '
                         'use rectified pinhole images and their corresponding intrinsics.')


def load_cameras_json(path):
    """Read OpenCV c2w poses and pixel intrinsics, indexed by image filename stem."""
    with Path(path).open(encoding='utf-8') as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError('Camera JSON must be an object containing extrinsics and intrinsics lists.')
    result = []
    for key, validator in [('extrinsics', validate_pose), ('intrinsics', validate_intrinsics)]:
        entries = data.get(key)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f'Camera JSON requires a non-empty {key} list.')
        matrices = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f'Each {key} entry must contain camera_id and matrix.')
            camera_id = entry.get('camera_id')
            if not isinstance(camera_id, str) or not camera_id:
                raise ValueError(f'Each {key} camera_id must be a non-empty string filename stem.')
            if camera_id in matrices:
                raise ValueError(f'Duplicate camera_id {camera_id!r} in {key}.')
            try:
                matrix = np.asarray(entry['matrix'], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Camera {camera_id}: {key} requires a numeric matrix.') from exc
            validator(matrix, camera_id)
            matrices[camera_id] = matrix
        result.append(matrices)
    return tuple(result)


def validate_cameras_for_images(images, poses, intrinsics):
    """Check coverage and principal points against the original, unprocessed images."""
    missing = [p.name for p in images if p.stem not in poses or p.stem not in intrinsics]
    if missing:
        raise ValueError(f'Missing camera poses/intrinsics for {len(missing)} images: '
                         f'{", ".join(missing[:5])}. Every input image needs both entries.')
    for path in images:
        validate_pose(poses[path.stem], path.stem)
        matrix = intrinsics[path.stem]
        validate_intrinsics(matrix, path.stem)
        with Image.open(path) as image:
            width, height = image.size
        if not (0 < matrix[0, 2] < width and 0 < matrix[1, 2] < height):
            raise ValueError(f'Camera {path.stem}: principal point must lie inside the original '
                             f'{width}x{height} image. Supply pixel intrinsics at this resolution.')


def write_cameras_json(path, images, poses, intrinsics, overwrite=False):
    """Validate all inputs before writing the JSON consumed by inference.py."""
    validate_cameras_for_images(images, poses, intrinsics)
    data = {'num_cameras': len(images), 'extrinsics': [], 'intrinsics': []}
    for image in images:
        for key, matrices in [('extrinsics', poses), ('intrinsics', intrinsics)]:
            data[key].append({'camera_id': image.stem, 'matrix': matrices[image.stem].tolist()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('w' if overwrite else 'x', encoding='utf-8') as handle:
            handle.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
    except FileExistsError as exc:
        raise FileExistsError(f'{path} already exists. Use --overwrite to replace it.') from exc
