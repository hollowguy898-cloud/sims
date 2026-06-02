"""
Spatial hash grid for O(1) neighbor lookups.

Replaces O(n²) distance scans with O(k) per-agent lookups where k
is the average number of agents per cell. With cell_size = max_vision_radius,
each agent only checks its own cell and 8 neighbors.
"""
import numpy as np
from typing import Generic, TypeVar

T = TypeVar('T')


class SpatialGrid(Generic[T]):
    """
    Spatial hash grid that maps grid cells to lists of entities.

    Cell size should be >= the maximum query radius for best performance.
    """

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[T]] = {}

    def clear(self):
        self.cells.clear()

    def insert(self, entity: T, x: float, y: float):
        """Insert an entity at position (x, y)."""
        cx = int(x // self.cell_size)
        cy = int(y // self.cell_size)
        key = (cx, cy)
        if key not in self.cells:
            self.cells[key] = []
        self.cells[key].append(entity)

    def query_radius_squared(self, x: float, y: float,
                              radius_sq: float) -> list[T]:
        """
        Return all entities within radius (squared) of (x, y).

        Uses squared distance to avoid sqrt.
        """
        radius = radius_sq ** 0.5  # only need this once for cell range
        min_cx = int((x - radius) // self.cell_size)
        max_cx = int((x + radius) // self.cell_size)
        min_cy = int((y - radius) // self.cell_size)
        max_cy = int((y + radius) // self.cell_size)

        result = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                bucket = self.cells.get((cx, cy))
                if bucket:
                    for entity in bucket:
                        result.append(entity)
        return result

    def query_radius(self, x: float, y: float, radius: float) -> list[T]:
        """Return all entities within radius of (x, y)."""
        return self.query_radius_squared(x, y, radius * radius)

    def build(self, entities: list, get_x, get_y):
        """Bulk-build the grid from a list of entities."""
        self.clear()
        for e in entities:
            self.insert(e, get_x(e), get_y(e))
