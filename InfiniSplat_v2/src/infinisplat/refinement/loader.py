"""Strict loading of the separate, inference-only refiner bundle."""
import json
from pathlib import Path
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file
from ..pipeline import sha256

BASE_SHA256 = '6d23f9cef753fb7f72ecd88fa8e074a458de5116591276da85129909e73a66e8'
CONTRACT = dict(
    format_version=1, architecture='fixed_center_knn_v1',
    checkpoint_identity='infinisplat_v2_refiner_v1',
    base_weights_sha256=BASE_SHA256,
    normalization='mean_predicted_point_radius', merge='uniform_mean',
    voxel_multiplier=1.4, opacity_threshold=.01,
    feature_dim=8, hidden_dim=64, knn_neighbors=8, knn_blocks=4,
    knn_radius=2, knn_fallback_radius=4,
    knn_interior_points=200000, knn_max_chunk_points=300000,
    recurrent_state_dim=16, recurrent_state_delta_max=.1,
    update_position=False, update_opacity=True, update_rotation=True, update_sh=True,
    scale_log_delta_max=0., opacity_logit_delta_max=.1, rotation_delta_max=.25,
    sh_logit_delta_max=1., gradient_input_mode='robust',
    default_steps=3, feedback_views_per_step=8, feedback_l1_weight=.8,
    feedback_dssim_weight=.2, ssim_window_size=11, sigma_extent=3.,
    rasterizer=dict(eps2d=.01, packed=True, absgrad=False,
                    rasterize_mode='classic', render_mode='RGB+ED', sh_degree=0),
)


def _matches_contract(value, expected):
    if isinstance(expected, dict):
        return (isinstance(value, dict) and value.keys() == expected.keys()
                and all(_matches_contract(value[k], v) for k, v in expected.items()))
    if isinstance(expected, float):
        return type(value) in (int, float) and value == expected
    return type(value) is type(expected) and value == expected


def validate_config(config, base_config):
    expected_keys = set(CONTRACT) | {'weights_sha256', 'source_version'}
    if (not isinstance(config, dict) or set(config) != expected_keys
            or any(not _matches_contract(config[k], v) for k, v in CONTRACT.items())):
        raise ValueError('Unsupported refiner architecture or inference contract')
    if (not isinstance(base_config, dict) or base_config.get('format_version') != 2 or base_config.get('backend') != 'da3'
            or base_config.get('weights_sha256') != BASE_SHA256
            or any(base_config.get(k) != CONTRACT[k] for k in
                   ('normalization', 'merge', 'voxel_multiplier', 'opacity_threshold'))
            or base_config.get('eps2d') != .01):
        raise ValueError('Refiner requires the exact released v2 base weights and configuration')
    digest = config['weights_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid refiner weight SHA256')
    if config['source_version'] != 'fixed_center_knn_v1':
        raise ValueError('Unsupported refiner inference source version')


def load_refiner(directory, base_config, device):
    path = Path(directory)
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(directory), allow_patterns=['config.json', 'refiner.safetensors']))
    config = json.loads((path / 'config.json').read_text())
    validate_config(config, base_config)
    if sha256(path / 'refiner.safetensors') != config['weights_sha256']:
        raise ValueError('Refiner weight integrity check failed')
    from .feedback import require_renderer
    require_renderer()
    from .refiner import RefinerBundle
    # This can be called from an inference_mode-decorated application.
    with torch.inference_mode(False), torch.no_grad():
        bundle = RefinerBundle()
        try:
            state = load_file(str(path / 'refiner.safetensors'))
        except SafetensorError as error:
            raise ValueError('Invalid refiner tensor file') from error
        expected = bundle.state_dict()
        if set(state) != set(expected) or any(
                state[k].shape != expected[k].shape or state[k].dtype != torch.float32
                or not torch.isfinite(state[k]).all() for k in expected):
            raise ValueError('Refiner tensor keys, shapes, dtype or values do not match')
        bundle.load_state_dict(state, strict=True)
        bundle.eval().requires_grad_(False).to(device)
    return bundle, config
