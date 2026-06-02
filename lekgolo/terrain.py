"""
Terrain generation for the Lekgolo simulation.
Uses Perlin-like noise for organic-looking terrain.
"""
import numpy as np
from config import (
    MAP_WIDTH, MAP_HEIGHT, TERRAIN_FLAT, TERRAIN_ROUGH, TERRAIN_WALL,
    TERRAIN_ROUGH_PROB, TERRAIN_WALL_PROB
)


def generate_terrain(width: int = MAP_WIDTH, height: int = MAP_HEIGHT,
                     seed: int | None = None) -> np.ndarray:
    """
    Generate a terrain grid.

    Returns:
        np.ndarray of shape (height, width) with values:
            TERRAIN_FLAT (0)  - normal movement
            TERRAIN_ROUGH (1) - slowed movement
            TERRAIN_WALL (2)  - impassable
    """
    rng = np.random.default_rng(seed)
    terrain = np.full((height, width), TERRAIN_FLAT, dtype=np.int8)

    # Generate smooth noise for rough terrain
    # Use a low-res noise and upscale for organic clusters
    noise_scale = 8
    low_h = height // noise_scale + 1
    low_w = width // noise_scale + 1
    noise = rng.random((low_h, low_w))

    # Bilinear upscale
    y_indices = np.linspace(0, low_h - 1, height)
    x_indices = np.linspace(0, low_w - 1, width)
    y0 = np.floor(y_indices).astype(int)
    y1 = np.minimum(y0 + 1, low_h - 1)
    x0 = np.floor(x_indices).astype(int)
    x1 = np.minimum(x0 + 1, low_w - 1)
    yf = y_indices - y0
    xf = x_indices - x0

    # Interpolate
    yf_2d = yf[:, None]
    xf_2d = xf[None, :]
    upscaled = (
        noise[y0[:, None], x0[None, :]] * (1 - yf_2d) * (1 - xf_2d) +
        noise[y0[:, None], x1[None, :]] * (1 - yf_2d) * xf_2d +
        noise[y1[:, None], x0[None, :]] * yf_2d * (1 - xf_2d) +
        noise[y1[:, None], x1[None, :]] * yf_2d * xf_2d
    )

    # Assign rough terrain where noise exceeds threshold
    rough_threshold = 1.0 - TERRAIN_ROUGH_PROB
    terrain[upscaled > rough_threshold] = TERRAIN_ROUGH

    # Wall clusters - use separate noise
    wall_noise = rng.random((low_h, low_w))
    upscaled_wall = (
        wall_noise[y0[:, None], x0[None, :]] * (1 - yf_2d) * (1 - xf_2d) +
        wall_noise[y0[:, None], x1[None, :]] * (1 - yf_2d) * xf_2d +
        wall_noise[y1[:, None], x0[None, :]] * yf_2d * (1 - xf_2d) +
        wall_noise[y1[:, None], x1[None, :]] * yf_2d * xf_2d
    )
    wall_threshold = 1.0 - TERRAIN_WALL_PROB
    wall_mask = upscaled_wall > wall_threshold
    # Walls don't overlap rough terrain
    terrain[wall_mask & (terrain == TERRAIN_FLAT)] = TERRAIN_WALL

    # Ensure borders have some open space for Flood spawning
    terrain[0:2, :] = TERRAIN_FLAT
    terrain[-2:, :] = TERRAIN_FLAT
    terrain[:, 0:2] = TERRAIN_FLAT
    terrain[:, -2:] = TERRAIN_FLAT

    return terrain


def is_passable(terrain: np.ndarray, x: int, y: int) -> bool:
    """Check if a cell is passable (not a wall and within bounds)."""
    h, w = terrain.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return terrain[y, x] != TERRAIN_WALL


def movement_cost(terrain: np.ndarray, x: int, y: int) -> float:
    """Get the movement multiplier at a position (higher = more costly)."""
    from config import TERRAIN_ROUGH_SPEED_PENALTY
    h, w = terrain.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return float('inf')
    if terrain[y, x] == TERRAIN_WALL:
        return float('inf')
    if terrain[y, x] == TERRAIN_ROUGH:
        return 1.0 + TERRAIN_ROUGH_SPEED_PENALTY
    return 1.0
