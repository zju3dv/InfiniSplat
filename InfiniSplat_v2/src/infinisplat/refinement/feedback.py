"""Context-only differentiable rendering feedback for frozen-network inference."""
import torch
import torch.nn.functional as F
from .parameters import quaternion_multiply, rotation_vector_to_quaternion, RGB2SH

def conservative_gaussian_frustum(
    means: torch.Tensor,
    scales: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    sigma_extent: float = 3.0,
    near: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scale-aware center-frustum mask and approximate pixel radius."""
    if means.numel() == 0:
        return (
            torch.zeros(means.shape[0], device=means.device, dtype=torch.bool),
            torch.zeros(means.shape[0], device=means.device, dtype=torch.float32),
        )
    with torch.no_grad(), torch.autocast(device_type=means.device.type, enabled=False):
        xyz = means.detach().float()
        radius = float(sigma_extent) * scales.detach().float().amax(dim=-1)
        world_to_camera = torch.linalg.inv(camera_to_world.detach().float())
        camera = xyz @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = camera[:, 2]
        safe_z = z.clamp_min(float(near))
        K = intrinsics.detach().float()
        u = camera[:, 0] / safe_z * K[0, 0] + K[0, 2]
        v = camera[:, 1] / safe_z * K[1, 1] + K[1, 2]
        # z-radius is the closest part of the bounding sphere and gives a conservative
        # image-space radius for Gaussians close to the camera.
        radius_depth = (z - radius).clamp_min(float(near))
        pixel_radius = radius * torch.maximum(K[0, 0], K[1, 1]) / radius_depth
        finite = (
            torch.isfinite(camera).all(dim=-1)
            & torch.isfinite(radius)
            & torch.isfinite(pixel_radius)
        )
        intersects = (
            finite
            & (z + radius > float(near))
            & (u + pixel_radius >= 0.0)
            & (u - pixel_radius <= float(width - 1))
            & (v + pixel_radius >= 0.0)
            & (v - pixel_radius <= float(height - 1))
        )
    return intersects, pixel_radius.clamp_min(0.0)

class Feedback:
    ssim_window_size = 11

    @staticmethod
    def _masked_l1(rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(rendered.dtype)[..., None]
        return ((rendered - target).abs() * weights).sum() / (
            weights.sum().clamp_min(1) * rendered.shape[-1]
        )

    def _masked_dssim(self, rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_nchw = mask.float()[None, None]
        left = rendered.permute(2, 0, 1).unsqueeze(0) * mask_nchw
        right = target.permute(2, 0, 1).unsqueeze(0) * mask_nchw
        window = min(self.ssim_window_size, rendered.shape[0], rendered.shape[1])
        window = max(3, window - 1 + window % 2)
        coordinate = torch.arange(window, device=rendered.device, dtype=rendered.dtype)
        gaussian = torch.exp(-((coordinate - (window - 1) / 2).square()) / (2 * 1.5**2))
        gaussian /= gaussian.sum()
        channels = rendered.shape[-1]
        kernel_x = gaussian.view(1, 1, 1, window).expand(channels, 1, 1, window)
        kernel_y = gaussian.view(1, 1, window, 1).expand(channels, 1, window, 1)

        def blur(value: torch.Tensor) -> torch.Tensor:
            radius = window // 2
            value = F.pad(value, (radius, radius, 0, 0), mode="reflect")
            value = F.conv2d(value, kernel_x, groups=channels)
            value = F.pad(value, (0, 0, radius, radius), mode="reflect")
            return F.conv2d(value, kernel_y, groups=channels)

        mean_left, mean_right = blur(left), blur(right)
        var_left = (blur(left.square()) - mean_left.square()).clamp_min(0)
        var_right = (blur(right.square()) - mean_right.square()).clamp_min(0)
        covariance = blur(left * right) - mean_left * mean_right
        ssim = ((2 * mean_left * mean_right + 0.01**2) * (2 * covariance + 0.03**2)) / (
            (mean_left.square() + mean_right.square() + 0.01**2)
            * (var_left + var_right + 0.03**2)
        ).clamp_min(1.0e-12)
        return (((1 - ssim) / 2) * mask_nchw).sum() / (
            mask_nchw.sum().clamp_min(1) * channels
        )

    @staticmethod
    def _robust_gradient_features(
        gradient_sum: torch.Tensor,
        gradient_square_sum: torch.Tensor,
        count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = count.clamp_min(1).unsqueeze(-1)
        mean = gradient_sum / denominator
        rms = (gradient_square_sum / denominator + 1.0e-12).sqrt()
        visible = count > 0
        consistency = mean / (rms + 1.0e-12)
        positive = rms[visible]
        reference = (
            positive.median(dim=0).values.detach()
            if positive.numel()
            else rms.new_ones((rms.shape[-1],))
        )
        strength = torch.log1p(rms / (reference + 1.0e-12))
        consistency[~visible] = 0
        strength[~visible] = 0
        return consistency.detach(), strength.detach()

    def gradients(self, scene, log_scale, opacity_logit, rgb_logit, image, camera, K):
        mask, _ = conservative_gaussian_frustum(scene['means'], scene['scales'], camera, K,
                                                *image.shape[-2:])
        ids = mask.nonzero(as_tuple=True)[0]
        if not ids.numel():
            return ids, None
        with torch.enable_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            scale = log_scale[ids].detach().clone().requires_grad_(True)
            opacity = opacity_logit[ids].detach().clone().requires_grad_(True)
            rotation = torch.zeros_like(scene['means'][ids], requires_grad=True)
            rgb = rgb_logit[ids].detach().clone().requires_grad_(True)
            quaternion = F.normalize(scene['quats'][ids].detach().float(), dim=-1)
            quaternion = F.normalize(quaternion_multiply(
                rotation_vector_to_quaternion(rotation), quaternion), dim=-1)
            colors = scene['sh'][ids].detach().float().clone()
            colors[:, 0] = colors[:, 0] + (RGB2SH(rgb.sigmoid())
                - RGB2SH(rgb_logit[ids].detach().float().sigmoid()))
            # Match the evaluated feedback ABI, including expected depth output.
            from gsplat import rasterization
            rendered, _, _ = rasterization(means=scene['means'][ids].detach().float(),
                quats=quaternion, scales=scale.exp(), opacities=opacity[:, 0].sigmoid(),
                colors=colors, viewmats=torch.linalg.inv(camera.detach().float()[None]),
                Ks=K.detach().float()[None], width=image.shape[-1], height=image.shape[-2],
                sh_degree=0, packed=True, absgrad=False, sparse_grad=False,
                rasterize_mode='classic', distributed=False, camera_model='pinhole',
                with_eval3d=False, render_mode='RGB+ED', eps2d=.01)
            target = image.detach().float().permute(1, 2, 0)
            valid = torch.ones_like(target[..., 0], dtype=torch.bool)
            loss = (.8 * self._masked_l1(rendered[0, ..., :3].float(), target, valid)
                    + .2 * self._masked_dssim(rendered[0, ..., :3].float(), target, valid))
            gradients = torch.autograd.grad(loss, (scale, opacity, rotation, rgb))
        if not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Non-finite renderer feedback gradient')
        return ids, gradients

    def __call__(self, scene, log_scale, opacity_logit, rgb_logit, images, cameras, K):
        count = log_scale.new_zeros(len(log_scale))
        sums = [torch.zeros_like(x) for x in
                (log_scale, opacity_logit, scene['means'], rgb_logit)]
        squares = [torch.zeros_like(x) for x in sums]
        for view in range(min(8, len(images))):
            ids, gradients = self.gradients(scene, log_scale, opacity_logit, rgb_logit,
                                            images[view], cameras[view], K[view])
            if gradients is None:
                continue
            count.index_add_(0, ids, count.new_ones(len(ids)))
            for total, square, gradient in zip(sums, squares, gradients):
                total.index_add_(0, ids, gradient)
                square.index_add_(0, ids, gradient.square())
        return [self._robust_gradient_features(total, square, count)
                for total, square in zip(sums, squares)]


def require_renderer():
    try:
        from gsplat import rasterization
    except ImportError as error:
        raise RuntimeError('Refinement requires the optional renderer: pip install ".[refiner]"') from error
    return rasterization
