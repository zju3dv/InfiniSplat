# Modified for SplatAnything: inference-only release.
import torch
import numpy as np
from typing import Tuple

def depth_to_camera_coords(depthmap, camera_intrinsics):
    """
    Convert depth map to 3D camera coordinates.
    
    Args:
        depthmap (BxHxW tensor): Batch of depth maps
        camera_intrinsics (Bx3x3 tensor): Camera intrinsics matrix for each camera
        
    Returns:
        X_cam (BxHxWx3 tensor): 3D points in camera coordinates
        valid_mask (BxHxW tensor): Mask indicating valid depth pixels
    """
    (B, H, W) = depthmap.shape
    device = depthmap.device
    dtype = depthmap.dtype
    camera_intrinsics = camera_intrinsics.float()
    fx = camera_intrinsics[:, 0, 0]
    fy = camera_intrinsics[:, 1, 1]
    cx = camera_intrinsics[:, 0, 2]
    cy = camera_intrinsics[:, 1, 2]
    (v_grid, u_grid) = torch.meshgrid(torch.arange(H, dtype=dtype, device=device), torch.arange(W, dtype=dtype, device=device), indexing='ij')
    u_grid = u_grid.unsqueeze(0)
    v_grid = v_grid.unsqueeze(0)
    z_cam = depthmap
    x_cam = (u_grid - cx.view(B, 1, 1)) * z_cam / fx.view(B, 1, 1)
    y_cam = (v_grid - cy.view(B, 1, 1)) * z_cam / fy.view(B, 1, 1)
    X_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    valid_mask = depthmap > 0.0
    return (X_cam, valid_mask)

def depth_to_world_coords_points(depth_map: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor, eps=1e-08) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of depth maps to world coordinates.

    Args:
        depth_map (torch.Tensor): (B, H, W) Depth map
        extrinsic (torch.Tensor): (B, 4, 4) Camera extrinsic matrix (camera-to-world transformation)
        intrinsic (torch.Tensor): (B, 3, 3) Camera intrinsic matrix

    Returns:
        world_coords_points (torch.Tensor): (B, H, W, 3) World coordinates
        camera_points (torch.Tensor): (B, H, W, 3) Camera coordinates
        point_mask (torch.Tensor): (B, H, W) Valid depth mask
    """
    if depth_map is None:
        return (None, None, None)
    point_mask = depth_map > eps
    (camera_points, _) = depth_to_camera_coords(depth_map, intrinsic)
    R_cam_to_world = extrinsic[:, :3, :3]
    t_cam_to_world = extrinsic[:, :3, 3]
    world_coords_points = torch.einsum('bhwi,bji->bhwj', camera_points, R_cam_to_world) + t_cam_to_world[:, None, None, :]
    return (world_coords_points, camera_points, point_mask)
