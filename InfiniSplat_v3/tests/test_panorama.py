"""Projection geometry and complete scene conversion, without model weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.utils.camera_io import list_image_files, load_cameras_json, validate_cameras_for_images
from src.utils.panorama import convert_panorama, perspective_view, view_cameras


class PanoramaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, path=None):
        path = path or self.root / 'pano.png'
        Image.new('RGB', (128, 64), (130, 90, 40)).save(path)
        return path

    def test_center_rays_follow_opencv_yaw_pitch_convention(self):
        height, width = 256, 512
        longitude = ((np.arange(width) + .5) / width - .5) * 2 * np.pi
        latitude = (.5 - (np.arange(height) + .5) / height) * np.pi
        lon, lat = np.meshgrid(longitude, latitude)
        directions = np.stack((np.sin(lon) * np.cos(lat), -np.sin(lat), np.cos(lon) * np.cos(lat)), axis=-1)
        panorama = np.rint((directions + 1) * 127.5).astype(np.uint8)
        for name, yaw, pitch, pose, intrinsic in view_cameras(33):
            with self.subTest(yaw=yaw, pitch=pitch):
                rgb = perspective_view(panorama, pose, intrinsic, 33)[16, 16]
                np.testing.assert_allclose(rgb, (pose[:3, 2] + 1) * 127.5, atol=1)

    def test_longitude_wraps_with_bilinear_sampling(self):
        panorama = np.zeros((8, 16, 3), dtype=np.uint8)
        panorama[:, 0, 2] = 255
        panorama[:, -1, 0] = 255
        _, _, _, pose, intrinsic = view_cameras(3)[4]  # 180 degrees, exactly on the seam
        np.testing.assert_allclose(perspective_view(panorama, pose, intrinsic, 3)[1, 1], [128, 0, 128], atol=1)

    def test_24_views_and_calibration_match_bundled_layout(self):
        cameras = view_cameras(2044)
        data = json.loads((Path(__file__).resolve().parents[1] / 'examples/pano/prior_camera.json').read_text())
        self.assertEqual(len(cameras), 24)
        for index, (name, yaw, pitch, pose, intrinsic) in enumerate(cameras):
            self.assertEqual(name, data['extrinsics'][index]['camera_id'])
            np.testing.assert_allclose(pose, data['extrinsics'][index]['matrix'], atol=1e-7)
            np.testing.assert_allclose(intrinsic, data['intrinsics'][index]['matrix'], rtol=1e-6)
        np.testing.assert_allclose(view_cameras(100, 90)[0][4], [[50, 0, 49.5], [0, 50, 49.5], [0, 0, 1]])

    def test_scene_is_ready_for_inference_and_source_is_preserved(self):
        output = self.root / 'scene'
        output.mkdir()
        source = self.source(output / 'pano.png')
        original = source.read_bytes()
        (output / 'notes.txt').write_text('keep me')
        convert_panorama(source, output, size=14)
        images = list_image_files(output / 'images')
        self.assertEqual(len(images), 24)
        self.assertEqual(images[-1].name, 'frame_0023.png')
        poses, intrinsics = load_cameras_json(output / 'prior_camera.json')
        validate_cameras_for_images(images, poses, intrinsics)
        metadata = json.loads((output / 'panorama.json').read_text())
        self.assertEqual(metadata['views'][16]['pitch_degrees'], 45)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual((output / 'notes.txt').read_text(), 'keep me')

    def test_overwrite_replaces_managed_files_only(self):
        source, output = self.source(), self.root / 'scene'
        convert_panorama(source, output, size=14)
        (output / 'images/stale.png').touch()
        (output / 'notes.txt').write_text('keep')
        with self.assertRaises(FileExistsError):
            convert_panorama(source, output, size=28)
        convert_panorama(source, output, size=28, overwrite=True)
        self.assertFalse((output / 'images/stale.png').exists())
        self.assertEqual((output / 'notes.txt').read_text(), 'keep')
        with Image.open(output / 'images/frame_0000.png') as image:
            self.assertEqual(image.size, (28, 28))
        with self.assertRaisesRegex(ValueError, 'outside'):
            convert_panorama(output / 'images/frame_0000.png', output, overwrite=True)

    def test_failed_install_restores_previous_scene(self):
        source, output = self.source(), self.root / 'scene'
        convert_panorama(source, output, size=14)
        before = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
        rename = Path.rename
        failed = False

        def fail_once(path, target):
            nonlocal failed
            if not failed and path.name == 'prior_camera.json' and path.parent != output:
                failed = True
                raise OSError('simulated install failure')
            return rename(path, target)

        with patch.object(Path, 'rename', fail_once), self.assertRaisesRegex(OSError, 'simulated'):
            convert_panorama(source, output, size=28, overwrite=True)
        after = {p.relative_to(output): p.read_bytes() for p in output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_invalid_source_and_projection_parameters(self):
        source = self.root / 'partial.png'
        Image.new('RGB', (64, 64)).save(source)
        with self.assertRaisesRegex(ValueError, '2:1'):
            convert_panorama(source, self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())
        for size, fov in [(0, 60), (1, 60), (14, 0), (14, 180), (14, float('nan'))]:
            with self.subTest(size=size, fov=fov), self.assertRaises(ValueError):
                view_cameras(size, fov)

    def test_converter_imports_without_torch(self):
        result = subprocess.run([sys.executable, '-c',
                                 'import src.utils.panorama; import sys; assert "torch" not in sys.modules'],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
