# Modified for SplatAnything: inference-only release.
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

def save_image_png(path: Path, image_tensor: torch.Tensor) -> None:
    # image_tensor: [H, W, 3]
    img = (image_tensor.detach().cpu() * 255.0).to(torch.uint8).numpy()
    Image.fromarray(img).save(str(path))

def save_depth_png(path: Path, depth_tensor: torch.Tensor) -> None:
    # depth_tensor: [H, W]
    d = depth_tensor.detach()
    d = d - d.min()
    d = d / (d.max() + 1e-9)
    img = (d.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(img, mode="L").save(str(path))

def save_depth_npy(path: Path, depth_tensor: torch.Tensor) -> None:
    # depth_tensor: [H, W]
    # Save actual depth values in numpy format
    d = depth_tensor.detach().cpu().numpy()
    np.save(str(path), d)

def _build_vertex_ply_element(pts: np.ndarray, colors: np.ndarray) -> PlyElement:
    """Build a PLY vertex element from points and colors arrays.

    Args:
        pts: Point coordinates, shape [N, 3], dtype float32
        colors: RGB colors, shape [N, 3], dtype uint8

    Returns:
        PlyElement describing the vertices
    """
    vertex_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex_elements = np.empty(len(pts), dtype=vertex_dtype)
    vertex_elements["x"] = pts[:, 0]
    vertex_elements["y"] = pts[:, 1]
    vertex_elements["z"] = pts[:, 2]
    vertex_elements["red"] = colors[:, 0]
    vertex_elements["green"] = colors[:, 1]
    vertex_elements["blue"] = colors[:, 2]
    return PlyElement.describe(vertex_elements, "vertex")

def save_scene_ply(path: Path,
                   points_xyz: torch.Tensor,
                   point_colors: torch.Tensor,
                   valid_mask: torch.Tensor = None) -> None:
    """Save point cloud to PLY format"""
    pts = points_xyz.detach().cpu().to(torch.float32).numpy().reshape(-1, 3)
    colors = point_colors.detach().cpu().to(torch.uint8).numpy().reshape(-1, 3)

    # Filter out invalid points (NaN, Inf)
    if valid_mask is None:
        valid_mask = np.isfinite(pts).all(axis=1)
    else:
        valid_mask = valid_mask.detach().cpu().numpy().reshape(-1)
    pts = pts[valid_mask]
    colors = colors[valid_mask]

    # Handle empty point cloud
    if len(pts) == 0:
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        colors = np.array([[255, 255, 255]], dtype=np.uint8)

    PlyData([_build_vertex_ply_element(pts, colors)]).write(str(path))

def _build_gs_ply_data(means, scales, rotations, rgbs, opacities, quantile_threshold):
    """Build Gaussian splat PLY data with scale-based filtering.

    Args:
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
        quantile_threshold: Percentile threshold for scale filtering (e.g. 0.98 or 0.90).
            None -> no filtering, export ALL Gaussians (inspect the full cloud).

    Returns:
        PlyData object ready to be written or returned
    """
    # None -> keep every Gaussian (no scale-based pruning); else drop the largest-scale tail.
    if quantile_threshold is None:
        filter_mask = torch.ones(scales.shape[0], dtype=torch.bool, device=scales.device)
    else:
        scale_threshold = torch.quantile(scales.max(dim=-1)[0], quantile_threshold, dim=0)
        filter_mask = scales.max(dim=-1)[0] <= scale_threshold

    means = means[filter_mask].reshape(-1, 3)
    scales = scales[filter_mask].reshape(-1, 3)
    rotations = rotations[filter_mask].reshape(-1, 4)
    rgbs = rgbs[filter_mask].reshape(-1, 3)
    opacities = opacities[filter_mask].reshape(-1)

    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")

    dtype_full = [(attribute, "f4") for attribute in attributes]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    # 3DGS / SuperSplat PLY storage convention (see process_ply_to_splat):
    #   color   = 0.5 + C0 * f_dc      -> store the SH DC coefficient (rgbs arg)
    #   opacity = sigmoid(opacity)     -> store the logit (inverse_sigmoid)
    #   scale   = exp(scale)           -> store log(scale)
    # rgbs is the SH DC coefficient (NOT activated RGB); callers pass sh[..., 0, :].
    opacity_logit = torch.logit(opacities.clamp(1e-6, 1 - 1e-6))
    attributes_data = (
        means.float().detach().cpu().numpy(),
        torch.zeros_like(means).float().detach().cpu().numpy(),
        rgbs.detach().cpu().contiguous().numpy(),
        opacity_logit[..., None].detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
    )
    attributes_data = np.concatenate(attributes_data, axis=1)
    elements[:] = list(map(tuple, attributes_data))

    return PlyData([PlyElement.describe(elements, "vertex")])

def save_gs_ply(path: Path,
                means: torch.Tensor,
                scales: torch.Tensor,
                rotations: torch.Tensor,
                rgbs: torch.Tensor,
                opacities: torch.Tensor,
                quantile_threshold: float = 0.98) -> None:
    """
    Export Gaussian splat data to PLY format.

    Args:
        path: Output PLY file path
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        rotations: Gaussian rotations as quaternions [N, 4]
        rgbs: RGB colors [N, 3]
        opacities: Opacity values [N]
        quantile_threshold: scale-tail pruning quantile (default 0.98 drops the largest 2%
            of Gaussians by scale). Pass None to export the FULL cloud with no filtering.
    """
    # Ensure float32 for quantile and numpy conversion (bf16 not supported)
    means, scales, rotations, rgbs, opacities = (
        t.float() for t in (means, scales, rotations, rgbs, opacities)
    )
    plydata = _build_gs_ply_data(means, scales, rotations, rgbs, opacities,
                                 quantile_threshold=quantile_threshold)
    plydata.write(str(path))
