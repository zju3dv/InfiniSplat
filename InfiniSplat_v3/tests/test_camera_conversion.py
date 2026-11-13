"""Camera conversion geometry, source-format parsing, and inference compatibility."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.convert_cameras import main
from src.utils.camera_conversion import cameras_from_arrays, cameras_from_colmap, to_opencv_c2w
from src.utils.camera_io import (
    list_image_files, load_cameras_json, validate_cameras_for_images, write_cameras_json,
)
from src.utils.scene_io import build_views


class CameraConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images_dir = self.root / 'images'
        self.images_dir.mkdir()
        for name in ('000.png', '001.png'):
            Image.new('RGB', (84, 56)).save(self.images_dir / name)
        self.images = list_image_files(self.images_dir)
        self.intrinsic = np.array([[70., 0, 42], [0, 80., 28], [0, 0, 1.]])
        angle = np.pi / 2
        self.pose = np.array([[np.cos(angle), -np.sin(angle), 0, 1],
                              [np.sin(angle), np.cos(angle), 0, 2],
                              [0, 0, 1, 3], [0, 0, 0, 1.]])

    def array_files(self, poses, intrinsics=None, names=('000.png', '001.png')):
        np.save(self.root / 'poses.npy', poses)
        np.save(self.root / 'intrinsics.npy', self.intrinsic if intrinsics is None else intrinsics)
        (self.root / 'names.txt').write_text('\n'.join(names) + '\n')
        return (self.root / 'poses.npy', self.root / 'intrinsics.npy', self.root / 'names.txt')

    def test_pose_direction_and_axis_conversion_preserve_world_points(self):
        flip = np.diag([1., -1., -1., 1.])
        point_cv = np.array([0.2, 0.3, 2, 1])
        expected_world = self.pose @ point_cv
        for convention in ('c2w', 'w2c'):
            for axes in ('opencv', 'opengl'):
                source_c2w = self.pose if axes == 'opencv' else self.pose @ flip
                source = source_c2w if convention == 'c2w' else np.linalg.inv(source_c2w)
                before = source.copy()
                converted = to_opencv_c2w(source, convention, axes)
                np.testing.assert_allclose(converted @ point_cv, expected_world, atol=1e-12)
                np.testing.assert_array_equal(source, before)

    def test_array_names_define_order_and_shared_intrinsics(self):
        second = self.pose.copy()
        second[0, 3] = 9
        files = self.array_files(np.stack([second, self.pose])[:, :3], names=('001.png', '000.png'))
        poses, intrinsics = cameras_from_arrays(self.images, *files, 'c2w', 'opencv')
        np.testing.assert_allclose(poses['000'], self.pose)
        np.testing.assert_allclose(poses['001'], second)
        np.testing.assert_allclose(intrinsics['001'], self.intrinsic)
        output = self.root / 'prior_camera.json'
        write_cameras_json(output, self.images, poses, intrinsics)
        views, _, _ = build_views(str(self.images_dir), str(output), (28, 28), cam_align=False)
        np.testing.assert_allclose(views['camera_poses'][0].numpy(), np.stack([self.pose, second]), atol=1e-6)

    def test_arrays_accept_per_view_intrinsics_and_colmap_pixel_centers(self):
        matrices = np.stack([self.intrinsic, self.intrinsic])
        matrices[:, :2, 2] += 0.5
        files = self.array_files(np.stack([self.pose, self.pose]), matrices)
        _, intrinsics = cameras_from_arrays(self.images, *files, 'c2w', 'opencv', 'colmap')
        np.testing.assert_allclose(intrinsics['000'], self.intrinsic)

    def test_array_name_count_and_coverage_errors(self):
        files = self.array_files(np.stack([self.pose, self.pose]), names=('000.png',))
        with self.assertRaisesRegex(ValueError, '1 names but poses has 2 rows'):
            cameras_from_arrays(self.images, *files, 'c2w', 'opencv')
        files = self.array_files(np.stack([self.pose, self.pose]), names=('000.png', 'missing.png'))
        with self.assertRaisesRegex(ValueError, 'no calibration'):
            cameras_from_arrays(self.images, *files, 'c2w', 'opencv')

    def colmap_model(self, model_format, camera_model='PINHOLE', size=(84, 56), name='001.png'):
        folder = self.root / model_format
        folder.mkdir(exist_ok=True)
        # A 90-degree z rotation in Hamilton WXYZ order, with a nonzero w2c translation.
        quaternion = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        translation = [1., 2., 3.]
        parameters = [70., 80., 42.5, 28.5]
        if camera_model == 'SIMPLE_PINHOLE':
            parameters = [70., 42.5, 28.5]
        elif camera_model == 'SIMPLE_RADIAL':
            parameters = [70., 42.5, 28.5, 0.01]
        if model_format == 'txt':
            (folder / 'cameras.txt').write_text(
                f'# cameras\n7 {camera_model} {size[0]} {size[1]} ' + ' '.join(map(str, parameters)) + '\n')
            (folder / 'images.txt').write_text(
                '# images\n3 1 0 0 0 0 0 0 7 000.png\n\n11 '
                + ' '.join(map(str, quaternion + translation)) + f' 7 {name}\n0 0 -1\n')
        else:
            model_id = {'SIMPLE_PINHOLE': 0, 'PINHOLE': 1, 'SIMPLE_RADIAL': 2}[camera_model]
            with (folder / 'cameras.bin').open('wb') as handle:
                handle.write(struct.pack('<QiiQQ', 1, 7, model_id, *size))
                handle.write(struct.pack('<' + 'd' * len(parameters), *parameters))
            with (folder / 'images.bin').open('wb') as handle:
                handle.write(struct.pack('<Q', 2))
                for image_id, filename, q, t, observations in [
                    (3, '000.png', [1., 0, 0, 0], [0., 0, 0], []),
                    (11, name, quaternion, translation, [(0., 0., -1)]),
                ]:
                    handle.write(struct.pack('<i7di', image_id, *q, *t, 7))
                    handle.write(filename.encode('utf-8') + b'\0')
                    handle.write(struct.pack('<Q', len(observations)))
                    for observation in observations:
                        handle.write(struct.pack('<ddq', *observation))
        return folder

    def test_colmap_text_and_binary_produce_identical_camera_geometry(self):
        expected = np.array([[0., 1., 0, -2], [-1., 0., 0, 1], [0., 0., 1, -3], [0, 0, 0, 1]])
        converted = []
        for model_format in ('txt', 'bin'):
            folder = self.colmap_model(model_format)
            poses, intrinsics = cameras_from_colmap(self.images, folder)
            np.testing.assert_allclose(poses['001'], expected, atol=1e-12)
            np.testing.assert_allclose(intrinsics['001'], self.intrinsic)
            converted.append((poses, intrinsics))
        for index in range(2):
            for key in ('000', '001'):
                np.testing.assert_allclose(converted[0][index][key], converted[1][index][key])

    def test_simple_pinhole_and_filename_with_spaces(self):
        self.images[1].rename(self.images_dir / 'frame with spaces.png')
        images = list_image_files(self.images_dir)
        for model_format in ('txt', 'bin'):
            folder = self.colmap_model(model_format, 'SIMPLE_PINHOLE', name='frame with spaces.png')
            poses, intrinsics = cameras_from_colmap(images, folder)
            self.assertIn('frame with spaces', poses)
            self.assertEqual(intrinsics['000'][1, 1], 70)

    def test_distorted_models_and_mismatched_resolution_are_rejected(self):
        for model_format in ('txt', 'bin'):
            folder = self.colmap_model(model_format, 'SIMPLE_RADIAL')
            with self.assertRaisesRegex(ValueError, 'image_undistorter'):
                cameras_from_colmap(self.images, folder)
        folder = self.colmap_model('txt', size=(168, 112))
        with self.assertRaisesRegex(ValueError, 'differs from calibration size'):
            cameras_from_colmap(self.images, folder)

    def test_truncated_binary_has_actionable_error(self):
        folder = self.colmap_model('bin')
        path = folder / 'images.bin'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            cameras_from_colmap(self.images, folder)

    def test_json_validation_rejects_bad_matrices_missing_names_and_duplicates(self):
        poses = {p.stem: np.eye(4) for p in self.images}
        intrinsics = {p.stem: self.intrinsic.copy() for p in self.images}
        output = self.root / 'prior_camera.json'
        write_cameras_json(output, self.images, poses, intrinsics)
        valid = json.loads(output.read_text())
        for kind in ('duplicate', 'rotation', 'intrinsics', 'missing'):
            data = json.loads(json.dumps(valid))
            if kind == 'duplicate':
                data['extrinsics'].append(data['extrinsics'][0])
            elif kind == 'rotation':
                data['extrinsics'][0]['matrix'][0][0] = 2
            elif kind == 'intrinsics':
                data['intrinsics'][0]['matrix'][0][1] = 0.1
            else:
                data['intrinsics'].pop()
            output.write_text(json.dumps(data))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                extr, intr = load_cameras_json(output)
                validate_cameras_for_images(self.images, extr, intr)

    def test_cli_roundtrip_check_and_overwrite_behavior(self):
        files = self.array_files(np.stack([np.eye(4), self.pose]))
        argv = ['arrays', '--images', str(self.images_dir), '--poses', str(files[0]),
                '--intrinsics', str(files[1]), '--image_names', str(files[2]),
                '--pose_convention', 'c2w', '--camera_axes', 'opencv']
        with contextlib.redirect_stdout(io.StringIO()):
            main(argv)
            main(['check', '--images', str(self.images_dir), '--cameras', str(self.root / 'prior_camera.json')])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(argv)
        with contextlib.redirect_stdout(io.StringIO()):
            main([*argv, '--overwrite'])

    def test_converter_import_does_not_require_torch_or_onnxruntime(self):
        root = Path(__file__).resolve().parents[1]
        code = ('import sys; from scripts.convert_cameras import main; '
                'assert "torch" not in sys.modules; assert "onnxruntime" not in sys.modules')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=root, check=True)


if __name__ == '__main__':
    unittest.main()
