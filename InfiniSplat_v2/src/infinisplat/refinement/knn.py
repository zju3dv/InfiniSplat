"""Fixed KNN attention and dependency-closure chunking; PyTorch only."""
from itertools import product
import torch
from torch import nn

class KNNTransformerBlock(nn.Module):
    """Single-head local attention over a precomputed KNN graph."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.scale = channels**-0.5

    def forward(self, features: torch.Tensor, knn: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(features)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        neighbor_key = key[knn.long()]
        neighbor_value = value[knn.long()]
        score = (query[:, None] * neighbor_key).sum(dim=-1) * self.scale
        context = (score.softmax(dim=-1)[..., None] * neighbor_value).sum(dim=1)
        features = features + self.proj(context)
        return features + self.mlp(self.norm2(features))

class ReSplatKNNRefinementNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(59, 64), nn.SiLU())
        self.blocks = nn.ModuleList(KNNTransformerBlock(64) for _ in range(4))
        self.output_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 26))

    def forward(self, features, knn):
        fine = self.input_projection(features)
        context = fine
        for block in self.blocks:
            context = block(context, knn)
        return self.output_head(torch.cat((fine, context), dim=-1))

def _voxel_knn_query(
    grid: torch.Tensor,
    points: torch.Tensor,
    query_ids: torch.Tensor,
    radius: int,
    neighbors: int,
    query_points: int,
) -> torch.Tensor:
    """Query occupied neighboring voxels without a quadratic point scan."""
    count = int(grid.shape[0])
    if query_ids.numel() == 0:
        return torch.empty((0, neighbors), device=grid.device, dtype=torch.long)
    minimum = grid.amin(dim=0) - int(radius)
    shifted = grid - minimum
    dimensions = shifted.amax(dim=0) + int(radius) + 1
    strides = torch.stack(
        (dimensions[1] * dimensions[2], dimensions[2], dimensions.new_tensor(1))
    )
    keys = (shifted * strides).sum(dim=-1)
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    offsets = torch.tensor(
        list(product(range(-radius, radius + 1), repeat=3)),
        device=grid.device,
        dtype=torch.long,
    )
    result = torch.full(
        (query_ids.numel(), neighbors), -1, device=grid.device, dtype=torch.long
    )
    for start in range(0, int(query_ids.numel()), int(query_points)):
        end = min(start + int(query_points), int(query_ids.numel()))
        ids = query_ids[start:end]
        candidate_grid = shifted[ids, None, :] + offsets[None, :, :]
        inside = ((candidate_grid >= 0) & (candidate_grid < dimensions)).all(dim=-1)
        candidate_keys = (candidate_grid * strides).sum(dim=-1)
        positions = torch.searchsorted(sorted_keys, candidate_keys)
        in_range = positions < count
        safe = positions.clamp_max(count - 1)
        candidate_ids = order[safe]
        valid = inside & in_range & (sorted_keys[safe] == candidate_keys)
        valid &= candidate_ids != ids[:, None]
        distances = (points[candidate_ids] - points[ids, None]).square().sum(dim=-1)
        distances = distances.masked_fill(~valid, float("inf"))
        take = min(neighbors, int(offsets.shape[0]))
        nearest_dist, nearest_local = torch.topk(
            distances, k=take, dim=-1, largest=False
        )
        nearest = candidate_ids.gather(1, nearest_local)
        nearest = nearest.masked_fill(~torch.isfinite(nearest_dist), -1)
        result[start:end, :take] = nearest
    return result

def voxel_knn_indices(
    grid: torch.Tensor,
    points: torch.Tensor,
    neighbors: int = 8,
    radius: int = 2,
    fallback_radius: int = 4,
    query_points: int = 20_000,
) -> torch.Tensor:
    """Build a local KNN graph on a one-point-per-voxel grid."""
    count = int(grid.shape[0])
    if count == 0:
        return torch.empty((0, neighbors), device=grid.device, dtype=torch.long)
    grid = grid.detach().long()
    points = points.detach().float()
    query_ids = torch.arange(count, device=grid.device)
    result = _voxel_knn_query(
        grid, points, query_ids, int(radius), int(neighbors), int(query_points)
    )
    unresolved = (result < 0).any(dim=-1)
    if int(fallback_radius) > int(radius) and bool(unresolved.any()):
        fallback_ids = unresolved.nonzero(as_tuple=True)[0]
        result[fallback_ids] = _voxel_knn_query(
            grid,
            points,
            fallback_ids,
            int(fallback_radius),
            int(neighbors),
            int(query_points),
        )
    self_ids = query_ids[:, None].expand_as(result)
    return torch.where(result >= 0, result, self_ids)

def knn_dependency_chunks(
    grid: torch.Tensor,
    knn: torch.Tensor,
    blocks: int = 4,
    interior_limit: int = 100_000,
    total_limit: int = 150_000,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Partition core points with the dependency closure for all KNN blocks."""
    count = int(grid.shape[0])
    if count == 0:
        return []
    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []

    def closure(interior: torch.Tensor) -> torch.Tensor:
        active = torch.unique(interior, sorted=True)
        for _ in range(int(blocks)):
            active = torch.unique(
                torch.cat((active, knn[active].reshape(-1))), sorted=True
            )
        return active

    def recurse(interior: torch.Tensor) -> None:
        active = closure(interior)
        if interior.numel() <= int(interior_limit) and active.numel() <= int(
            total_limit
        ):
            local = torch.searchsorted(active, interior)
            if not torch.equal(active[local], interior):
                raise RuntimeError("KNN refinement chunk lost an interior point")
            chunks.append((active, local))
            return
        if interior.numel() <= 1:
            raise RuntimeError(
                f"KNN closure contains {active.numel()} points, above {total_limit}"
            )
        span = grid[interior].amax(dim=0) - grid[interior].amin(dim=0)
        axis = int(span.argmax())
        order = torch.argsort(grid[interior, axis], stable=True)
        middle = int(interior.numel()) // 2
        recurse(interior[order[:middle]])
        recurse(interior[order[middle:]])

    ids = torch.arange(count, device=grid.device)
    if ids.numel() <= int(interior_limit):
        recurse(ids)
    else:
        span = grid.amax(dim=0) - grid.amin(dim=0)
        axis = int(span.argmax())
        order = torch.argsort(grid[:, axis], stable=True)
        for start in range(0, count, int(interior_limit)):
            recurse(order[start : start + int(interior_limit)])
    return chunks
