"""
Terrain utilities - thin wrapper around world_gen for backward compatibility.
Actual terrain generation is now in world_gen.py using the full pipeline.
"""
from world_gen import is_passable, movement_cost, generate_heightmap, heightmap_to_terrain
from config import (
    MAP_WIDTH, MAP_HEIGHT, TERRAIN_FLAT, TERRAIN_ROUGH, TERRAIN_WALL
)


def generate_terrain(width: int = MAP_WIDTH, height: int = MAP_HEIGHT,
                     seed: int | None = None) -> 'np.ndarray':
    """
    Generate terrain using the new heightmap-based pipeline.
    Kept for backward compatibility with any code that calls this directly.
    """
    import numpy as np
    heightmap = generate_heightmap(width, height, seed)
    terrain = heightmap_to_terrain(heightmap)
    return terrain
