"""CPU regression checks requiring no model weights."""
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from plyfile import PlyData
from unittest.mock import patch

from validation import load_config, main, parse_args
from inference import resolve_scene_inputs
from src.models.models.rasterization import GaussianSplatRenderer, DPTDecoder
from src.models.models.visual_transformer import VisualGeometryTransformer
from src.utils.checkpoints import load_model_weights
from src.utils.scene_io import build_views, load_cameras_json, preprocess_view, export_gaussians, export_points


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def images(self):
        folder = self.root / 'scene/images'
        folder.mkdir(parents=True)
        for name, value in [('001.JPG', 100), ('000.png', 10), ('002.webp', 200)]:
            Image.fromarray(np.full((56, 84, 3), value, np.uint8)).save(folder / name)
        return folder

    def test_all_images_are_sorted_and_loaded_in_rgb_mode(self):
        folder = self.images()
        for index in range(3, 67):
            Image.new('RGB', (28, 28), color=(index, index, index)).save(folder / f'{index:03d}.png')
        views, name, paths = build_views(str(folder), resolution_wh=(28, 28))
        self.assertEqual(name, 'scene')
        self.assertEqual([Path(p).stem for p in paths], [f'{index:03d}' for index in range(67)])
        self.assertEqual(views['img'].shape, (1, 67, 3, 28, 28))
        self.assertNotIn('camera_poses', views)

    def test_resize_preserves_aspect_and_updates_intrinsics(self):
        image = Image.fromarray(np.zeros((56, 84, 3), np.uint8))
        matrix = np.array([[100, 0, 42], [0, 100, 28], [0, 0, 1]], np.float32)
        rgb, intrinsic = preprocess_view(image, matrix, (28, 28))
        self.assertEqual(rgb.shape, (3, 28, 28))
        self.assertAlmostEqual(intrinsic[0, 0].item(), 50, places=5)
        self.assertAlmostEqual(intrinsic[1, 1].item(), 50, places=5)

    def test_camera_matching_and_first_view_alignment(self):
        folder = self.images()
        identity = np.eye(4)
        identity[0, 3] = 10
        other = identity.copy()
        other[0, 3] = 12
        last = identity.copy()
        last[0, 3] = 14
        camera = self.root / 'cameras.json'
        matrix = [[100, 0, 42], [0, 100, 28], [0, 0, 1]]
        cameras = dict(
            extrinsics=[dict(camera_id=i, matrix=pose.tolist()) for i, pose in
                        [('000', identity), ('001', other), ('002', last)]],
            intrinsics=[dict(camera_id=i, matrix=matrix) for i in ['000', '001', '002']])
        camera.write_text(json.dumps(cameras))
        views, _, _ = build_views(str(folder), str(camera), (28, 28))
        torch.testing.assert_close(views['camera_poses'][0, :, 0, 3], torch.tensor([0., 2., 4.]))
        cameras['extrinsics'].pop()
        camera.write_text(json.dumps(cameras))
        with self.assertRaisesRegex(ValueError, '002'):
            build_views(str(folder), str(camera), (28, 28))

    def test_cli_validation(self):
        self.assertEqual(parse_args(['--input_path', '.']).resolution, [756, 756])
        for extra in [['--resolution', '100', '98'], ['--sky_max_fraction', '2']]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--input_path', '.', '--output_path', '.', *extra])

    def test_scene_paths_work_for_examples_and_custom_data(self):
        for folder in ['examples/object', 'data/my scene', str(self.root / 'custom-scene')]:
            for mode in ['rgb', 'pose']:
                with self.subTest(folder=folder, mode=mode):
                    args = parse_args(['--input_path', folder, '--mode', mode,
                                       '--resolution', '518', '518',
                                       '--output_path', 'my-results'])
                    self.assertEqual(args.input_path, folder)
                    self.assertEqual(args.pose_condition, mode == 'pose')
                    self.assertEqual(args.resolution, [518, 518])
                    self.assertEqual(args.output_path, 'my-results')

    def test_cli_requires_input_and_rejects_conflicting_modes(self):
        for argv in [[], ['--mode', 'rgb'], ['--input_path', '.', '--mode', 'invalid'],
                     ['--input_path', '.', '--mode', 'rgb', '--pose_condition']]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)

    def config_args(self, enabled=False, use_camera=False, extra=()):
        scene = self.root / 'scene'
        (scene / 'images').mkdir(parents=True, exist_ok=True)
        weights = self.root / 'placeholder.safetensors'
        weights.touch()
        config = self.root / 'config.yaml'
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/inference.yaml')
        cfg.wrapper.enable_cond = enabled
        cfg.wrapper.model.use_gt_camera = use_camera
        cfg.wrapper.SplatAnything_pretrained = str(weights)
        cfg.wrapper.worldmirror2_pretrained = str(weights)
        OmegaConf.save(cfg, config)
        return parse_args(['--config_path', str(config), '--input_path', str(scene), '--output_path', '.', *extra])

    def test_mode_sets_both_camera_flags_consistently(self):
        args = self.config_args(True, True, ['--mode', 'rgb'])
        camera = self.root / 'scene/prior_camera.json'
        camera.write_text('ignored in RGB mode')
        cfg = load_config(args)
        self.assertFalse(cfg.wrapper.enable_cond or cfg.wrapper.model.use_gt_camera)
        self.assertIsNone(args.cameras)
        args = self.config_args(extra=['--mode', 'pose'])
        cfg = load_config(args)
        self.assertTrue(cfg.wrapper.enable_cond and cfg.wrapper.model.use_gt_camera)
        self.assertEqual(args.cameras, str(camera))

    def test_pose_requires_scene_camera_and_rgb_ignores_it(self):
        with self.assertRaisesRegex(FileNotFoundError, 'prior_camera.json'):
            load_config(self.config_args(extra=['--mode', 'pose']))
        args = self.config_args(extra=['--mode', 'rgb'])
        load_config(args)
        self.assertIsNone(args.cameras)
        self.assertEqual(args.images_path, str(self.root / 'scene/images'))
        (self.root / 'scene/prior_camera.json').write_text('not valid JSON')
        load_config(args)
        self.assertIsNone(args.cameras)

    def test_scene_layout_errors_are_reported_before_weights_or_models_load(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Scene directory not found'):
            resolve_scene_inputs(self.root / 'missing', False)
        (self.root / 'empty').mkdir()
        with self.assertRaisesRegex(FileNotFoundError, 'images/'):
            resolve_scene_inputs(self.root / 'empty', False)
        folder = self.images()
        with self.assertRaisesRegex(FileNotFoundError, 'images/'):
            resolve_scene_inputs(folder, False)
        with patch('src.utils.checkpoints.load_model_weights') as weights:
            args = self.config_args(extra=['--mode', 'pose'])
            with self.assertRaisesRegex(FileNotFoundError, 'prior_camera.json'):
                main(['--config_path', args.config_path, '--input_path', str(folder.parent), '--mode', 'pose'])
            weights.assert_not_called()

    def test_pose_flags_must_be_paired_and_have_cameras(self):
        with self.assertRaisesRegex(ValueError, 'both'):
            load_config(self.config_args(True, False))
        with self.assertRaisesRegex(FileNotFoundError, 'prior_camera.json'):
            load_config(self.config_args(True, True))
        camera = self.root / 'scene/prior_camera.json'
        camera.write_text('{}')
        args = self.config_args(extra=['--pose_condition'])
        cfg = load_config(args)
        self.assertTrue(cfg.wrapper.enable_cond and cfg.wrapper.model.use_gt_camera)
        cfg = load_config(self.config_args(True, True, ['--no_pose_condition']))
        self.assertFalse(cfg.wrapper.enable_cond or cfg.wrapper.model.use_gt_camera)

    def test_checkpoint_override_keeps_sky_selection_in_cli(self):
        override = self.root / 'custom.pth'
        override.touch()
        args = self.config_args(extra=['--sky_seg', '--ckpt_path', str(override)])
        cfg = load_config(args)
        self.assertEqual(cfg.wrapper.SplatAnything_pretrained, str(override.resolve()))
        self.assertNotIn('stage1_gaussian_pretrained', cfg.wrapper)
        self.assertTrue(args.sky_seg)
        self.assertNotIn('filter_sky_infer', cfg.wrapper.model)

    def test_sky_cli_controls_segmentation_and_export_together(self):
        folder = self.images()

        class Teacher(torch.nn.Module):
            def forward(self, views, cond_flags):
                batch, count, _, height, width = views['img'].shape
                return dict(depth=torch.ones(batch, count, height, width, 1),
                            camera_poses=torch.eye(4).repeat(batch, count, 1, 1),
                            camera_intrs=torch.eye(3).repeat(batch, count, 1, 1))

        class Student(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.renderer = GaussianSplatRenderer(filter_sky_infer=kwargs['filter_sky_infer'])
                with torch.no_grad():
                    for parameter in self.parameters():
                        parameter.zero_()
                    self.renderer.decoder.gs_head[-1].bias[0] = 1

            def forward(self, views, geometry, cond_flags):
                batch, _, _, height, width = views['img'].shape
                return self.renderer(views, geometry, lambda start, end:
                                     torch.zeros(batch, end-start, 128, height, width))

        class Segmenter:
            def sky_mask(self, rgb):
                sky = np.zeros(rgb.shape[:2], dtype=bool)
                sky[:sky.shape[0] // 2] = True
                return sky

        for enabled in (False, True):
            output = self.root / f'output_{enabled}'
            extra = ['--input_path', str(folder.parent), '--output_path', str(output),
                     '--resolution', '28', '28',
                     '--device', 'cpu', '--no_depth_rgb']
            if enabled:
                extra.append('--sky_seg')
            args = self.config_args(extra=extra)
            argv = ['--config_path', args.config_path, *extra]
            with patch('src.models.models.gaussian_worldmirror.GaussianWorldMirror', side_effect=Student) as student, \
                 patch('src.models.models.geometry_worldmirror.GeometryWorldMirror', return_value=Teacher()), \
                 patch('src.utils.checkpoints.load_model_weights'), \
                 patch('src.utils.sky_segmentation.SkySegmenter', return_value=Segmenter()) as segmenter, \
                 contextlib.redirect_stdout(io.StringIO()):
                main(argv)
            self.assertEqual(student.call_args.kwargs['filter_sky_infer'], enabled)
            self.assertEqual(segmenter.call_count, int(enabled))
            points = PlyData.read(output / 'scene/points.ply')['vertex']
            self.assertEqual(len(points), 3 * 28 * 28 // (2 if enabled else 1))
            self.assertEqual(len(json.loads((output / 'scene/cameras.json').read_text())['extrinsics']), 3)
            saved_config = OmegaConf.load(output / 'scene/config.yaml')
            self.assertNotIn('filter_sky_infer', saved_config.wrapper.model)
            self.assertEqual(json.loads((output / 'scene/run.json').read_text())['sky_seg'], enabled)

    def test_minimal_config_has_only_public_fields(self):
        cfg = load_config(self.config_args())
        self.assertEqual(set(cfg.wrapper), {
            'worldmirror2_pretrained', 'SplatAnything_pretrained',
            'enable_cond', 'infer_opacity_thresh', 'model',
        })
        self.assertEqual(set(cfg.wrapper.model), {
            'use_gt_camera', 'filter_flying_points_infer',
            'filter_large_scale_infer', 'max_scale_abs', 'max_scale_pct', 'gs_voxel_size',
        })

    def test_internal_settings_are_not_exposed_in_yaml(self):
        for key in ('decode_chunk_size', 'spacing_target', 'filter_sky_train', 'filter_sky_infer'):
            args = self.config_args()
            cfg = OmegaConf.load(args.config_path)
            cfg.wrapper.model[key] = 8
            OmegaConf.save(cfg, args.config_path)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unsupported'):
                load_config(args)

    def test_documented_camera_json_matches_loader(self):
        guide = (Path(__file__).resolve().parents[1] / 'prepare_data.md').read_text()
        data = json.loads(re.search(r'```json\n(.*?)\n```', guide, re.S).group(1))
        path = self.root / 'prior_camera.json'
        path.write_text(json.dumps(data))
        poses, intrinsics = load_cameras_json(path)
        self.assertEqual(set(poses), {'000', '001'})
        self.assertEqual(set(intrinsics), {'000', '001'})
        # The documented c2w translation is the camera center, not a w2c translation.
        np.testing.assert_allclose(poses['001'][:3, 3], [1, 0, 0])
        np.testing.assert_allclose(intrinsics['000'], [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])

    def test_invalid_absolute_scale_threshold(self):
        args = self.config_args()
        cfg = OmegaConf.load(args.config_path)
        cfg.wrapper.model.max_scale_abs = -1
        OmegaConf.save(cfg, args.config_path)
        with self.assertRaisesRegex(ValueError, 'max_scale_abs'):
            load_config(args)

    def test_incomplete_checkpoint_is_rejected(self):
        layer = torch.nn.Linear(3, 2)
        checkpoint = self.root / 'partial.pth'
        torch.save({'weight': layer.weight}, checkpoint)
        with self.assertRaisesRegex(ValueError, 'missing'):
            load_model_weights(layer, checkpoint)

    def test_encoder_constructs_and_runs_both_condition_modes(self):
        for enabled in (False, True):
            encoder = VisualGeometryTransformer(img_size=28, patch_size=14, embed_dim=32,
                                                depth=1, num_heads=4, patch_embed='conv',
                                                enable_cond=enabled, normalized_rope=True,
                                                intermediate_idxs=[0]).eval()
            images = torch.rand(1, 2, 3, 28, 28)
            priors = (None, torch.rand(1, 2, 4), torch.rand(1, 2, 7)) if enabled else None
            flags = [1, 0, 1] if enabled else [0, 0, 0]
            with torch.inference_mode():
                tokens, start = encoder(images, priors, cond_flags=flags)
            self.assertEqual(start, 7 if enabled else 5)
            self.assertEqual(tokens[0].shape, (1, 2, start + 4, 64))

    def test_voxel_merge_preserves_source_rules(self):
        renderer = GaussianSplatRenderer(voxel_size=1)
        splats = dict(means=torch.tensor([[[0.1, 0., 0.], [0.3, 0., 0.], [2., 0., 0.]]]),
                      weights=torch.tensor([[1., 3., 2.]]),
                      quats=torch.tensor([[[1., 0., 0., 0.]] * 3]),
                      scales=torch.tensor([[[1., 1., 1.], [3., 3., 3.], [5., 5., 5.]]]),
                      opacities=torch.tensor([[0.01, 0.02, 0.03]]), sh=torch.ones(1, 3, 1, 3))
        merged = renderer.merge_voxels(splats, torch.ones(1, 3, dtype=torch.bool))
        torch.testing.assert_close(merged['means'][0][:, 0], torch.tensor([0.25, 2.]))
        torch.testing.assert_close(merged['scales'][0][:, 0], torch.tensor([2.5, 5.]))
        torch.testing.assert_close(merged['opacities'][0], torch.tensor([2.5, 2.]))
        self.assertEqual(renderer.merge_voxels(splats, torch.zeros(1, 3, dtype=torch.bool))['means'][0].shape, (0, 3))

    def test_gaussian_spacing_normalization_sets_spacing_and_preserves_geometry(self):
        renderer = GaussianSplatRenderer(spacing_target=0.125)
        depth = torch.tensor([4., 6.]).reshape(2, 1, 1, 1).repeat(1, 2, 2, 2)
        predicted = torch.eye(4).repeat(2, 2, 1, 1)
        predicted[:, 1, 0, 3] = torch.tensor([1., 2.])
        poses = torch.eye(4).repeat(2, 2, 1, 1)
        poses[:, :, 0, 3] = torch.tensor([[5., 8.], [10., 14.]])
        intrinsics = torch.eye(3).repeat(2, 2, 1, 1)
        intrinsics[0, :, 0, 0], intrinsics[0, :, 1, 1] = 2, 8
        intrinsics[1, :, 0, 0], intrinsics[1, :, 1, 1] = 9, 16
        originals = [value.clone() for value in (depth, poses, intrinsics, predicted)]

        normalized_depth, normalized_poses = renderer.normalize_gaussian_spacing(
            depth, poses, intrinsics, predicted)
        # Geometric-mean focals are 4 and 12; the target footprints imply depths 0.5 and 1.5.
        expected_depth = torch.tensor([0.5, 1.5]).reshape(2, 1, 1, 1).expand_as(depth)
        torch.testing.assert_close(normalized_depth, expected_depth)
        expected_poses = poses.clone()
        # Camera baseline alignment is 3x and 2x, giving scene scales 1/24 and 1/8.
        expected_poses[0, :, :3, 3] /= 24
        expected_poses[1, :, :3, 3] /= 8
        torch.testing.assert_close(normalized_poses, expected_poses)
        for actual, original in zip((depth, poses, intrinsics, predicted), originals):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)

        # Changing the input world unit must produce the same canonical geometry.
        scaled_poses, scaled_predicted = poses.clone(), predicted.clone()
        scaled_poses[..., :3, 3] *= 7
        scaled_predicted[..., :3, 3] *= 7
        rescaled_depth, rescaled_poses = renderer.normalize_gaussian_spacing(
            depth * 7, scaled_poses, intrinsics, scaled_predicted)
        torch.testing.assert_close(rescaled_depth, normalized_depth)
        torch.testing.assert_close(rescaled_poses, normalized_poses)

    def test_gaussian_spacing_normalization_uses_only_valid_pixels_for_scale(self):
        renderer = GaussianSplatRenderer(spacing_target=1)
        depth = torch.tensor([[[[2., 4.], [100., 200.]]]])
        poses = torch.eye(4).reshape(1, 1, 4, 4)
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3)
        valid = torch.tensor([[[[True, True], [False, False]]]])
        normalized_depth, _ = renderer.normalize_gaussian_spacing(
            depth, poses, intrinsics, poses, valid_mask=valid)
        # The valid median is 2; sky pixels share the resulting transform but do not set it.
        torch.testing.assert_close(normalized_depth, torch.tensor([[[[1., 2.], [50., 100.]]]]))
        unchanged_depth, unchanged_poses = renderer.normalize_gaussian_spacing(
            depth, poses, intrinsics, poses, valid_mask=torch.zeros_like(valid))
        torch.testing.assert_close(unchanged_depth, depth)
        torch.testing.assert_close(unchanged_poses, poses)

    def test_chunked_decode_agrees_with_unchunked(self):
        torch.manual_seed(3)
        decoder = DPTDecoder(decode_chunk_size=2).eval()
        feat = torch.randn(1, 3, 128, 4, 6)
        rgb = torch.rand(1, 3, 3, 4, 6)
        depth = torch.ones(1, 3, 4, 6)
        poses = torch.eye(4).repeat(1, 3, 1, 1)
        intr = torch.eye(3).repeat(1, 3, 1, 1)
        with torch.inference_mode():
            a = decoder(lambda s, e: feat[:, s:e], rgb, depth, poses, intr)
            decoder.decode_chunk_size = 10
            b = decoder(lambda s, e: feat[:, s:e], rgb, depth, poses, intr)
        for key in a:
            torch.testing.assert_close(a[key], b[key])

    def test_sky_filter_reaches_points_and_gaussians(self):
        renderer = GaussianSplatRenderer(max_scale_pct=None, voxel_size=0.00001,
                                          filter_sky_infer=True).eval()
        # Predict zero raw parameters: finite identity-normalized geometry and constant scales.
        for parameter in renderer.parameters():
            parameter.data.zero_()
        views = dict(img=torch.ones(1, 1, 3, 4, 4), depthmap=torch.zeros(1, 1, 4, 4))
        views['depthmap'][..., :2, :] = -1
        geometry = dict(depth=torch.ones(1, 1, 4, 4, 1),
                        camera_poses=torch.eye(4).reshape(1, 1, 4, 4),
                        camera_intrs=torch.eye(3).reshape(1, 1, 3, 3))
        with torch.inference_mode():
            out = renderer(views, geometry, lambda s, e: torch.zeros(1, 1, 128, 4, 4))
        self.assertEqual(len(out['points_ply_means'][0]), 8)
        self.assertEqual(len(out['splats']['means'][0]), 8)
        torch.testing.assert_close(out['export_camera_intrs'], geometry['camera_intrs'])
        renderer.filter_sky_infer = False
        with torch.inference_mode():
            unfiltered = renderer(views, geometry, lambda s, e: torch.zeros(1, 1, 128, 4, 4))
        self.assertEqual(len(unfiltered['points_ply_means'][0]), 16)
        self.assertEqual(len(unfiltered['splats']['means'][0]), 16)

    def render_filter_fixture(self, depth=None, **settings):
        height, width = (3, 6)
        n = height * width
        renderer = GaussianSplatRenderer(voxel_size=0.001, **settings).eval()
        images = torch.ones(1, 1, 3, height, width)
        if depth is None:
            depth = torch.ones(1, 1, height, width, 1)
        geometry = dict(depth=depth, camera_poses=torch.eye(4).reshape(1, 1, 4, 4),
                        camera_intrs=torch.eye(3).reshape(1, 1, 3, 3))
        scale = torch.arange(1, n + 1).float() / 100
        splats = dict(means=torch.arange(n).float().reshape(1, n, 1).expand(-1, -1, 3),
                      scales=scale.reshape(1, n, 1).expand(-1, -1, 3),
                      quats=torch.tensor([1., 0., 0., 0.]).repeat(1, n, 1),
                      opacities=torch.full((1, n), 0.5), weights=torch.full((1, n), 0.5),
                      sh=torch.zeros(1, n, 1, 3), base_rgb=torch.ones(1, n, 3))
        with patch.object(renderer.decoder, 'forward', return_value=splats), torch.inference_mode():
            return renderer(dict(img=images), geometry, features=None)

    def test_large_scale_switch_and_both_thresholds(self):
        cases = [
            (dict(filter_large_scale_infer=True, max_scale_abs=0.10, max_scale_pct=None), 10),
            (dict(filter_large_scale_infer=True, max_scale_abs=None, max_scale_pct=50), 9),
            (dict(filter_large_scale_infer=True, max_scale_abs=0.05, max_scale_pct=50), 5),
            (dict(filter_large_scale_infer=False, max_scale_abs=0.01, max_scale_pct=1), 18),
        ]
        for settings, expected in cases:
            with self.subTest(settings=settings):
                result = self.render_filter_fixture(**settings)
                self.assertEqual(len(result['points_ply_means'][0]), expected)
                self.assertEqual(len(result['splats']['means'][0]), expected)

    def test_flying_point_switch(self):
        depth = torch.ones(1, 1, 3, 6, 1)
        depth[:, :, :, 3:] = 2
        for enabled, expected in [(False, 18), (True, 12)]:
            result = self.render_filter_fixture(depth=depth, filter_flying_points_infer=enabled,
                                                max_scale_pct=None)
            self.assertEqual(len(result['points_ply_means'][0]), expected)
            self.assertEqual(len(result['splats']['means'][0]), expected)

    def test_ply_export_roundtrip(self):
        preds = dict(splats=dict(means=[torch.tensor([[1., 2., 3.]])],
                                scales=[torch.tensor([[0.1, 0.2, 0.3]])],
                                quats=[torch.tensor([[1., 0., 0., 0.]])],
                                opacities=[torch.tensor([0.5])], sh=[torch.zeros(1, 1, 3)]),
                     points_ply_means=[torch.tensor([[1., 2., 3.]])],
                     points_ply_rgb=[torch.tensor([[1., 0., 0.]])])
        export_gaussians(preds, self.root, 0.05, None)
        export_points(preds, self.root)
        vertex = PlyData.read(self.root / 'gaussians.ply')['vertex']
        self.assertAlmostEqual(float(vertex['scale_0'][0]), np.log(0.1), places=6)
        self.assertAlmostEqual(float(vertex['opacity'][0]), 0)
        self.assertEqual(int(PlyData.read(self.root / 'points.ply')['vertex']['red'][0]), 255)


if __name__ == '__main__':
    unittest.main()
