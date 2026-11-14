"""Inference-only, fixed-center recurrent Gaussian attribute refinement."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .knn import ReSplatKNNRefinementNetwork, voxel_knn_indices, knn_dependency_chunks
from .feedback import Feedback
from .parameters import quaternion_multiply, rotation_vector_to_quaternion, RGB2SH, SH2RGB


class AttributeRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ReSplatKNNRefinementNetwork()
        self.state_projection = nn.Linear(8, 16)

    def residual(self, parts, graph, chunks):
        result_ids, result_values = [], []
        for active, interior in chunks:
            features = torch.cat([value[active].float() for value in parts], dim=-1)
            neighbor_global = graph[active]
            local = torch.searchsorted(active, neighbor_global)
            safe = local.clamp_max(active.numel() - 1)
            valid = active[safe] == neighbor_global
            self_ids = torch.arange(active.numel(), device=active.device)[:, None]
            neighbors = torch.where(valid, safe, self_ids).long()
            values = self.network(features, neighbors)
            result_ids.append(active[interior])
            result_values.append(values[interior])
        ids, values = torch.cat(result_ids), torch.cat(result_values)
        order = torch.argsort(ids)
        if not torch.equal(ids[order], torch.arange(len(graph), device=graph.device)):
            raise RuntimeError('Refinement chunks must cover every Gaussian exactly once')
        values = values[order]
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite refinement residual')
        return values

    @torch.no_grad()
    def forward(self, scene, features, images, cameras, K, voxel_size, steps=3):
        size = float(voxel_size)
        if not math.isfinite(size) or size <= 0:
            raise ValueError('Refinement needs the native positive merge voxel size')
        if features.shape != (len(scene['means']), 8):
            raise ValueError('Refinement features do not match merged Gaussians')
        grid = torch.floor(scene['means'].detach().float() / size).long()
        grid = grid - grid.amin(dim=0, keepdim=True)
        graph = voxel_knn_indices(grid, scene['means'], neighbors=8, radius=2, fallback_radius=4)
        distances = (scene['means'][graph[:, :4]].detach().float()
                     - scene['means'].detach().float()[:, None]).norm(dim=-1)
        chunks = knn_dependency_chunks(grid, graph, blocks=4,
                                      interior_limit=200000, total_limit=300000)
        hidden = self.state_projection(features.float())
        feedback = Feedback()
        for _ in range(steps):
            scales = scene['scales'].detach().float()
            if not torch.isfinite(scales).all() or (scales <= 0).any():
                raise FloatingPointError('Gaussian scales must be finite and positive')
            log_scale = scales.clamp_min(1e-12).log()
            opacity = scene['opacities'].detach().float()
            opacity_logit = torch.logit(opacity.clamp(1e-6, 1-1e-6))[:, None]
            quaternion = F.normalize(scene['quats'].detach().float(), dim=-1)
            sh = scene['sh'].detach()
            rgb_logit = torch.logit(SH2RGB(sh[:, 0].float()).clamp(1e-6, 1-1e-6))
            (scale_c, scale_h), (opacity_c, opacity_h), (rotation_c, rotation_h), (rgb_c, rgb_h) = (
                feedback(scene, log_scale, opacity_logit, rgb_logit, images, cameras, K))
            parts = [features, torch.log(scales / size), quaternion, distances / size,
                     scale_c, scale_h, opacity_logit, opacity_c, opacity_h,
                     rotation_c, rotation_h, rgb_logit, rgb_c, rgb_h, hidden]
            residual = self.residual(parts, graph, chunks)
            # Zero scale_log_delta_max means an unbounded log-scale update.
            updated_scale = scales * torch.exp(residual[:, :3].float())
            decoded_opacity = torch.sigmoid((opacity_logit + .1 * residual[:, 3:4].tanh())[:, 0])
            updated_opacity = (opacity + decoded_opacity - opacity_logit[:, 0].sigmoid()).clamp(0, 1)
            rotation_delta = .25 * residual[:, 4:7].tanh()
            updated_quaternion = F.normalize(quaternion_multiply(
                rotation_vector_to_quaternion(rotation_delta), quaternion), dim=-1)
            updated_rgb_logit = rgb_logit + residual[:, 7:10].tanh()
            updated_sh = sh.clone()
            updated_sh[:, 0] = sh[:, 0] + (RGB2SH(updated_rgb_logit.sigmoid())
                                          - RGB2SH(rgb_logit.sigmoid())).to(sh.dtype)
            hidden = hidden + .1 * residual[:, 10:26].tanh()
            scene = dict(means=scene['means'].detach(), scales=updated_scale,
                         opacities=updated_opacity, quats=updated_quaternion, sh=updated_sh)
            if (not all(torch.isfinite(v).all() for v in scene.values())
                    or (updated_scale <= 0).any()
                    or (updated_quaternion.norm(dim=-1) <= 1e-8).any()):
                raise FloatingPointError('Invalid Gaussian attributes after refinement')
        return scene


class RefinerBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(128, 8)
        self.refiner = AttributeRefiner()

    @torch.no_grad()
    def project(self, features):
        # Called inside the base model's inference/autocast scope. The evaluated
        # projector consumes FP32 flattened decoder features outside autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            dense = features.float().flatten(0, 1).permute(0, 2, 3, 1).reshape(-1, 128)
            return self.projector(dense)
