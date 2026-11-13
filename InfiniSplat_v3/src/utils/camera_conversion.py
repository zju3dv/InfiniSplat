"""Convert existing calibration data into SplatAnything's camera JSON convention."""
from pathlib import Path
import os
import struct

import numpy as np
from PIL import Image

from src.utils.camera_io import validate_pose, validate_intrinsics
from src.utils.geometry import colmap_to_opencv_intrinsics


def to_opencv_c2w(matrix, pose_convention, camera_axes):
    """Invert w2c first, then convert camera axes while preserving the world frame."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    validate_pose(matrix, 'source')
    if pose_convention not in {'c2w', 'w2c'} or camera_axes not in {'opencv', 'opengl'}:
        raise ValueError('Specify c2w/w2c and opencv/opengl explicitly.')
    result = np.linalg.inv(matrix) if pose_convention == 'w2c' else matrix.copy()
    if camera_axes == 'opengl':
        result = result @ np.diag([1., -1., -1., 1.])
    return result


def _store(records, name, pose, intrinsic, image_size=None):
    # Inference takes a flat image directory. Never collapse ambiguous source names silently.
    basename = Path(name).name
    stem = Path(basename).stem
    if basename in records or any(Path(existing).stem == stem for existing in records):
        raise ValueError(f'Ambiguous or duplicate source image name: {name!r}. '
                         'Each image needs a unique filename and stem.')
    records[basename] = (pose, intrinsic, image_size)


def _match_records(images, records):
    missing = [p.name for p in images if p.name not in records]
    if missing:
        raise ValueError(f'{len(missing)} input images have no calibration: '
                         f'{", ".join(missing[:5])}. Keep only calibrated/registered images '
                         'in the input folder and preserve their filenames.')
    poses, intrinsics = {}, {}
    for path in images:
        pose, intrinsic, expected_size = records[path.name]
        if expected_size is not None:
            with Image.open(path) as image:
                actual_size = image.size
            if actual_size != expected_size:
                raise ValueError(f'{path.name}: image size {actual_size} differs from calibration '
                                 f'size {expected_size}. Use the matching undistorted images/model; '
                                 'do not mix original and resized calibration data.')
        poses[path.stem] = pose
        intrinsics[path.stem] = intrinsic
    return poses, intrinsics


def cameras_from_arrays(images, poses_path, intrinsics_path, image_names_path,
                        pose_convention, camera_axes, intrinsics_pixel_center='opencv'):
    """Match matrix rows to explicit image names; a single K may be shared by all views."""
    names = [line.strip() for line in Path(image_names_path).read_text(encoding='utf-8').splitlines()
             if line.strip()]
    poses = np.load(poses_path, allow_pickle=False)
    intrinsics = np.load(intrinsics_path, allow_pickle=False)
    if not isinstance(poses, np.ndarray) or not isinstance(intrinsics, np.ndarray):
        raise ValueError('Use numeric .npy arrays, not .npz archives or object arrays.')
    if poses.ndim == 2:
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] not in {(4, 4), (3, 4)}:
        raise ValueError('Poses must have shape [N, 4, 4] or [N, 3, 4].')
    if len(names) != len(poses):
        raise ValueError(f'image_names.txt has {len(names)} names but poses has {len(poses)} rows.')
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (len(poses), 3, 3))
    if intrinsics.shape != (len(poses), 3, 3):
        raise ValueError('Intrinsics must have shape [3, 3] or [N, 3, 3], in original-image pixels.')
    if intrinsics_pixel_center not in {'opencv', 'colmap'}:
        raise ValueError('Intrinsics pixel center must be opencv or colmap.')
    records = {}
    for name, pose, intrinsic in zip(names, poses, intrinsics):
        intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
        if intrinsics_pixel_center == 'colmap':
            intrinsic = colmap_to_opencv_intrinsics(intrinsic)
        validate_intrinsics(intrinsic, name)
        _store(records, name, to_opencv_c2w(pose, pose_convention, camera_axes), intrinsic)
    return _match_records(images, records)


def _pinhole_camera(model, width, height, params):
    if model == 'SIMPLE_PINHOLE' and len(params) == 3:
        fx, cx, cy = params
        fy = fx
    elif model == 'PINHOLE' and len(params) == 4:
        fx, fy, cx, cy = params
    else:
        raise ValueError(f'Unsupported COLMAP camera model/parameters: {model}. '
                         'Run COLMAP image_undistorter and use its PINHOLE or SIMPLE_PINHOLE '
                         'model together with the undistorted images.')
    if width <= 0 or height <= 0:
        raise ValueError('COLMAP camera image dimensions must be positive.')
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    # COLMAP locates the first pixel center at (0.5, 0.5); inference uses (0, 0).
    matrix = colmap_to_opencv_intrinsics(matrix)
    validate_intrinsics(matrix, 'COLMAP')
    return matrix, (width, height)


def _pose_from_colmap(qvec, tvec):
    """Convert COLMAP's Hamilton WXYZ quaternion and w2c translation to c2w."""
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError('COLMAP pose has an invalid quaternion.')
    w, x, y, z = qvec / norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = -rotation.T @ np.asarray(tvec, dtype=np.float64)
    validate_pose(matrix, 'COLMAP')
    return matrix


def _read_text_model(model_dir):
    cameras, records, image_ids = {}, {}, set()
    for line in (model_dir / 'cameras.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError('Malformed cameras.txt camera line.')
        camera_id = int(parts[0])
        if camera_id in cameras:
            raise ValueError(f'Duplicate COLMAP camera ID: {camera_id}')
        cameras[camera_id] = _pinhole_camera(parts[1], int(parts[2]), int(parts[3]),
                                             [float(value) for value in parts[4:]])
    with (model_dir / 'images.txt').open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.strip().split(maxsplit=9)
            if len(parts) != 10:
                raise ValueError('Malformed images.txt image header.')
            image_id = int(parts[0])
            if image_id in image_ids:
                raise ValueError(f'Duplicate COLMAP image ID: {image_id}')
            image_ids.add(image_id)
            camera_id = int(parts[8])
            if camera_id not in cameras:
                raise ValueError(f'COLMAP image {parts[9]} refers to missing camera {camera_id}.')
            matrix, size = cameras[camera_id]
            pose = _pose_from_colmap([float(v) for v in parts[1:5]], [float(v) for v in parts[5:8]])
            _store(records, parts[9], pose, matrix, size)
            points_line = handle.readline()
            if not points_line or len(points_line.split()) % 3:
                raise ValueError('Each images.txt image header must be followed by a POINTS2D line '
                                 '(an empty line is valid for zero observations).')
    return records


def _unpack(handle, layout):
    size = struct.calcsize('<' + layout)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f'Truncated COLMAP binary file: {handle.name}')
    return struct.unpack('<' + layout, data)


def _read_binary_model(model_dir):
    cameras, records, image_ids = {}, {}, set()
    with (model_dir / 'cameras.bin').open('rb') as handle:
        count, = _unpack(handle, 'Q')
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(handle, 'iiQQ')
            if model_id not in {0, 1}:
                raise ValueError(f'Unsupported COLMAP binary camera model ID {model_id}. '
                                 'Use image_undistorter to export a pinhole model and matching images.')
            if camera_id in cameras:
                raise ValueError(f'Duplicate COLMAP camera ID: {camera_id}')
            model, num_params = ('SIMPLE_PINHOLE', 3) if model_id == 0 else ('PINHOLE', 4)
            cameras[camera_id] = _pinhole_camera(model, width, height, _unpack(handle, 'd'*num_params))
    with (model_dir / 'images.bin').open('rb') as handle:
        count, = _unpack(handle, 'Q')
        for _ in range(count):
            values = _unpack(handle, 'i7di')
            image_id, camera_id = values[0], values[-1]
            if image_id in image_ids:
                raise ValueError(f'Duplicate COLMAP image ID: {image_id}')
            image_ids.add(image_id)
            name = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP image name.')
                if char == b'\0':
                    break
                name.extend(char)
            name = name.decode('utf-8')
            if camera_id not in cameras:
                raise ValueError(f'COLMAP image {name} refers to missing camera {camera_id}.')
            matrix, size = cameras[camera_id]
            _store(records, name, _pose_from_colmap(values[1:5], values[5:8]), matrix, size)
            points, = _unpack(handle, 'Q')
            end = handle.tell() + points * 24
            if end > os.fstat(handle.fileno()).st_size:
                raise ValueError(f'Truncated COLMAP point observations for image {name}.')
            handle.seek(end)
    return records


def cameras_from_colmap(images, model_dir, model_format='auto'):
    """Read the cameras/images pair only; points3D, pycolmap, and GPU libraries are not needed."""
    model_dir = Path(model_dir)
    if model_format not in {'auto', 'bin', 'txt'}:
        raise ValueError('COLMAP model format must be auto, bin, or txt.')
    if model_format == 'auto':
        model_format = next((ext for ext in ('bin', 'txt') if all(
            (model_dir / f'{name}.{ext}').is_file() for name in ('cameras', 'images'))), None)
    if model_format is None or not all((model_dir / f'{name}.{model_format}').is_file()
                                       for name in ('cameras', 'images')):
        raise ValueError(f'No matching cameras/images .bin or .txt pair found in {model_dir}. '
                         'Point --model at one sparse reconstruction, e.g. sparse/0.')
    records = _read_binary_model(model_dir) if model_format == 'bin' else _read_text_model(model_dir)
    return _match_records(images, records)
