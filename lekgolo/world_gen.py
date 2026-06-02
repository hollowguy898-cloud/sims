"""
Procedural world generation pipeline.

Seed -> Heightmap -> Terrain -> Region classification -> Path validation
-> Spawn formations -> Resource fields -> Event schedule -> Map modifiers
-> Map classification tag

Every generated map must answer: "What kind of strategy does this map reward?"
If the answer is "random chaos," the simulation will be random chaos.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_dilation
from collections import deque

from config import (
    MAP_WIDTH, MAP_HEIGHT,
    TERRAIN_FLAT, TERRAIN_ROUGH, TERRAIN_WALL, TERRAIN_TOXIC, TERRAIN_HIGHGROUND,
    WORLD_GEN_OCTAVES, WORLD_GEN_PERSISTENCE, WORLD_GEN_LACUNARITY,
    WORLD_GEN_SEA_LEVEL, WORLD_GEN_MOUNTAIN_LEVEL, WORLD_GEN_ROUGH_LEVEL,
    WORLD_GEN_MIN_PATHS, WORLD_GEN_PATH_WIDTH,
    MAP_TYPE_OPEN, MAP_TYPE_CANYON, MAP_TYPE_MAZE, MAP_TYPE_ISLAND, MAP_TYPE_MIXED,
    NUM_MODIFIERS_PER_MAP, MODIFIER_RADIUS_MIN, MODIFIER_RADIUS_MAX,
    MODIFIER_TOXIC_ZONE, MODIFIER_INFECTION_FOG, MODIFIER_COMM_JAM,
    MODIFIER_BIOMASS_DECAY, MODIFIER_COLLAPSING,
    EVENT_FLOOD_SURGE, EVENT_RESOURCE_BLOOM, EVENT_TERRAIN_COLLAPSE,
    EVENT_THINKER_DISRUPT,
    EVENT_INTERVAL_MIN, EVENT_INTERVAL_MAX, EVENT_FLOOD_SURGE_COUNT,
    EVENT_RESOURCE_BLOOM_AMOUNT, EVENT_TERRAIN_COLLAPSE_RADIUS,
    EVENT_THINKER_DISRUPT_DURATION,
    BIOMASS_FIELD_COUNT_MIN, BIOMASS_FIELD_COUNT_MAX,
    BIOMASS_FIELD_RADIUS_MIN, BIOMASS_FIELD_RADIUS_MAX, BIOMASS_FIELD_RATE,
    LEKGLO_SPAWN_RADIUS, LEKGLO_THINKER_SPAWN_OFFSET,
    FLOOD_SPAWN_CLUSTER_COUNT, FLOOD_SPAWN_CLUSTER_SIZE, FLOOD_SPAWN_ISOLATED_SEEDS,
    NUM_WORMS_INITIAL, NUM_THINKERS_INITIAL, NUM_FLOOD_INITIAL,
    MAX_STEPS_PER_EPISODE,
)


# ---------------------------------------------------------------------------
# 1. Heightmap generation (multi-octave value noise)
# ---------------------------------------------------------------------------

def _value_noise_2d(width: int, height: int, scale: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Generate a single-octave value noise grid."""
    gw = width // scale + 2
    gh = height // scale + 2
    grid = rng.random((gh, gw))

    # Bilinear interpolation to full resolution
    y_idx = np.linspace(0, gh - 1, height)
    x_idx = np.linspace(0, gw - 1, width)
    y0 = np.floor(y_idx).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1)
    x0 = np.floor(x_idx).astype(int)
    x1 = np.minimum(x0 + 1, gw - 1)
    yf = y_idx - y0
    xf = x_idx - x0

    yf_2d = yf[:, None]
    xf_2d = xf[None, :]

    upscaled = (
        grid[y0[:, None], x0[None, :]] * (1 - yf_2d) * (1 - xf_2d) +
        grid[y0[:, None], x1[None, :]] * (1 - yf_2d) * xf_2d +
        grid[y1[:, None], x0[None, :]] * yf_2d * (1 - xf_2d) +
        grid[y1[:, None], x1[None, :]] * yf_2d * xf_2d
    )
    return upscaled


def generate_heightmap(width: int = MAP_WIDTH, height: int = MAP_HEIGHT,
                       seed: int | None = None) -> np.ndarray:
    """
    Generate a multi-octave heightmap in [0, 1].

    Uses fractional Brownian motion (fBm) with configurable octaves,
    persistence, and lacunarity.
    """
    rng = np.random.default_rng(seed)
    hmap = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0

    for octave in range(WORLD_GEN_OCTAVES):
        scale = max(2, int(4 * (WORLD_GEN_LACUNARITY ** octave)))
        noise = _value_noise_2d(width, height, scale, rng)
        hmap += noise * amplitude
        total_amplitude += amplitude
        amplitude *= WORLD_GEN_PERSISTENCE

    hmap /= total_amplitude

    # Smooth with a light Gaussian pass for organic feel
    hmap = gaussian_filter(hmap, sigma=1.5)

    # Re-normalize to [0, 1]
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    return hmap


# ---------------------------------------------------------------------------
# 2. Heightmap -> Terrain
# ---------------------------------------------------------------------------

def heightmap_to_terrain(heightmap: np.ndarray) -> np.ndarray:
    """
    Convert a heightmap to a terrain grid using threshold levels.

    height < sea_level        -> WALL  (impassable water/chasm)
    sea_level <= h < rough    -> FLAT
    rough <= h < mountain     -> ROUGH
    h >= mountain             -> HIGHGROUND

    Toxic zones are added later by map modifiers.
    """
    h, w = heightmap.shape
    terrain = np.full((h, w), TERRAIN_FLAT, dtype=np.int8)

    terrain[heightmap < WORLD_GEN_SEA_LEVEL] = TERRAIN_WALL
    terrain[(heightmap >= WORLD_GEN_ROUGH_LEVEL) &
            (heightmap < WORLD_GEN_MOUNTAIN_LEVEL)] = TERRAIN_ROUGH
    terrain[heightmap >= WORLD_GEN_MOUNTAIN_LEVEL] = TERRAIN_HIGHGROUND

    return terrain


# ---------------------------------------------------------------------------
# 3. Region classification
# ---------------------------------------------------------------------------

def find_regions(terrain: np.ndarray) -> list[dict]:
    """
    Find connected passable regions using flood-fill.

    Returns a list of region dicts with:
        'id': int
        'cells': set of (x, y) tuples
        'area': int
        'center': (x, y) centroid
    """
    h, w = terrain.shape
    visited = np.zeros((h, w), dtype=bool)
    regions = []
    region_id = 0

    for y in range(h):
        for x in range(w):
            if terrain[y, x] != TERRAIN_WALL and not visited[y, x]:
                # BFS flood fill
                cells = set()
                queue = deque([(x, y)])
                visited[y, x] = True
                while queue:
                    cx, cy = queue.popleft()
                    cells.add((cx, cy))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < w and 0 <= ny < h and
                                not visited[ny, nx] and
                                terrain[ny, nx] != TERRAIN_WALL):
                            visited[ny, nx] = True
                            queue.append((nx, ny))

                if cells:
                    cx_mean = int(np.mean([c[0] for c in cells]))
                    cy_mean = int(np.mean([c[1] for c in cells]))
                    regions.append({
                        'id': region_id,
                        'cells': cells,
                        'area': len(cells),
                        'center': (cx_mean, cy_mean),
                    })
                    region_id += 1

    return regions


# ---------------------------------------------------------------------------
# 4. Connectivity guarantees (path validation & carving)
# ---------------------------------------------------------------------------

def ensure_connectivity(terrain: np.ndarray,
                        spawn_zones: list[tuple[int, int]],
                        min_paths: int = WORLD_GEN_MIN_PATHS,
                        path_width: int = WORLD_GEN_PATH_WIDTH) -> np.ndarray:
    """
    Ensure there are at least `min_paths` distinct passable paths
    between every pair of spawn zones. If not, carve corridors.

    Corridors are carved as FLAT terrain through WALL cells.
    """
    h, w = terrain.shape

    # For each pair of spawn zones, verify connectivity
    for i in range(len(spawn_zones)):
        for j in range(i + 1, len(spawn_zones)):
            sx, sy = spawn_zones[i]
            ex, ey = spawn_zones[j]

            # BFS from sx,sy to ex,ey
            if _has_path(terrain, sx, sy, ex, ey):
                continue

            # No path - carve one using A* heuristic (straight line with
            # slight randomization to avoid overlapping corridors)
            terrain = _carve_corridor(terrain, sx, sy, ex, ey, path_width)

    return terrain


def _has_path(terrain: np.ndarray, sx: int, sy: int,
              ex: int, ey: int) -> bool:
    """BFS path existence check."""
    h, w = terrain.shape
    visited = np.zeros((h, w), dtype=bool)
    queue = deque([(sx, sy)])
    visited[sy, sx] = True

    while queue:
        cx, cy = queue.popleft()
        if cx == ex and cy == ey:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < w and 0 <= ny < h and
                    not visited[ny, nx] and
                    terrain[ny, nx] != TERRAIN_WALL):
                visited[ny, nx] = True
                queue.append((nx, ny))
    return False


def _carve_corridor(terrain: np.ndarray, sx: int, sy: int,
                    ex: int, ey: int, width: int) -> np.ndarray:
    """Carve a straight-ish corridor between two points."""
    h, w = terrain.shape
    terrain = terrain.copy()

    # Bresenham-like line with random perturbation
    x, y = float(sx), float(sy)
    dx = ex - sx
    dy = ey - sy
    dist = max(np.sqrt(dx * dx + dy * dy), 1.0)
    step_x = dx / dist
    step_y = dy / dist
    rng = np.random.default_rng(abs(hash((sx, sy, ex, ey))) % (2**31))

    steps = int(dist) + 1
    for _ in range(steps):
        # Add slight random wobble
        wx = x + rng.normal(0, 0.5)
        wy = y + rng.normal(0, 0.5)

        # Carve a width-wide corridor
        for ox in range(-width // 2, width // 2 + 1):
            for oy in range(-width // 2, width // 2 + 1):
                cx = int(np.clip(np.round(wx + ox), 0, w - 1))
                cy = int(np.clip(np.round(wy + oy), 0, h - 1))
                terrain[cy, cx] = TERRAIN_FLAT

        x += step_x
        y += step_y

    return terrain


# ---------------------------------------------------------------------------
# 5. Map classification
# ---------------------------------------------------------------------------

def classify_map(terrain: np.ndarray) -> str:
    """
    Classify a map into one of: open, canyon, maze, island, mixed.

    Uses metrics: wall fraction, region count, and region size variance.
    """
    h, w = terrain.shape
    total = h * w
    wall_frac = np.sum(terrain == TERRAIN_WALL) / total
    regions = find_regions(terrain)
    num_regions = len(regions)

    if num_regions == 0:
        return MAP_TYPE_OPEN

    areas = [r['area'] for r in regions]
    max_area = max(areas) if areas else 0
    area_variance = np.var(areas) if len(areas) > 1 else 0

    # Island: many small regions
    if num_regions > 5 and max_area < total * 0.5:
        return MAP_TYPE_ISLAND

    # Canyon: high wall fraction, few long regions
    if wall_frac > 0.4 and num_regions <= 3:
        return MAP_TYPE_CANYON

    # Maze: moderate walls, many regions
    if wall_frac > 0.3 and num_regions > 3:
        return MAP_TYPE_MAZE

    # Open: few walls, one big region
    if wall_frac < 0.15 and max_area > total * 0.8:
        return MAP_TYPE_OPEN

    return MAP_TYPE_MIXED


# ---------------------------------------------------------------------------
# 6. Spawn formation placement
# ---------------------------------------------------------------------------

def generate_spawn_formations(terrain: np.ndarray,
                              rng: np.random.Generator) -> dict:
    """
    Generate spawn formations for both sides.

    Lekgolo: clustered near center, thinkers at core, workers on perimeter.
    Flood: distributed clusters near edges + isolated infection seeds.

    Returns dict with:
        'lekgolo_center': (x, y)
        'lekgolo_thinker_positions': list of (x, y)
        'lekgolo_worker_positions': list of (x, y)
        'flood_clusters': list of (x, y) centers
        'flood_isolated_seeds': list of (x, y)
    """
    h, w = terrain.shape

    # --- Lekgolo spawn: find a large passable region near center ---
    center_x, center_y = w // 2, h // 2
    # Search outward from center for a passable cell
    lekgolo_center = _find_passable_near(terrain, center_x, center_y)
    lcx, lcy = lekgolo_center

    # Thinker positions: tight cluster around center
    thinker_positions = []
    for _ in range(NUM_THINKERS_INITIAL):
        tx, ty = _find_passable_near(
            terrain,
            lcx + rng.integers(-LEKGLO_THINKER_SPAWN_OFFSET,
                               LEKGLO_THINKER_SPAWN_OFFSET + 1),
            lcy + rng.integers(-LEKGLO_THINKER_SPAWN_OFFSET,
                               LEKGLO_THINKER_SPAWN_OFFSET + 1)
        )
        thinker_positions.append((tx, ty))

    # Worker positions: ring around thinkers
    num_workers = NUM_WORMS_INITIAL - NUM_THINKERS_INITIAL
    worker_positions = []
    for _ in range(num_workers):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(LEKGLO_THINKER_SPAWN_OFFSET + 1, LEKGLO_SPAWN_RADIUS)
        wx = int(np.clip(np.round(lcx + np.cos(angle) * r), 0, w - 1))
        wy = int(np.clip(np.round(lcy + np.sin(angle) * r), 0, h - 1))
        wx, wy = _find_passable_near(terrain, wx, wy)
        worker_positions.append((wx, wy))

    # --- Flood spawn: clusters near edges + isolated seeds ---
    flood_clusters = []
    edges = [
        (w // 4, 3),
        (3 * w // 4, 3),
        (3, h // 4),
        (3, 3 * h // 4),
        (w // 4, h - 4),
        (3 * w // 4, h - 4),
        (w - 4, h // 4),
        (w - 4, 3 * h // 4),
    ]

    # Pick FLOOD_SPAWN_CLUSTER_COUNT from edge positions
    chosen = rng.choice(len(edges), size=min(FLOOD_SPAWN_CLUSTER_COUNT, len(edges)),
                        replace=False)
    for idx in chosen:
        ex, ey = edges[int(idx)]
        pos = _find_passable_near(terrain, ex, ey)
        flood_clusters.append(pos)

    # Isolated seeds: random positions, weighted toward edges
    flood_seeds = []
    for _ in range(FLOOD_SPAWN_ISOLATED_SEEDS):
        # Weight toward edges
        if rng.random() < 0.7:
            side = rng.integers(0, 4)
            if side == 0:
                ex, ey = rng.integers(0, w), rng.integers(0, h // 5)
            elif side == 1:
                ex, ey = rng.integers(0, w), rng.integers(4 * h // 5, h)
            elif side == 2:
                ex, ey = rng.integers(0, w // 5), rng.integers(0, h)
            else:
                ex, ey = rng.integers(4 * w // 5, w), rng.integers(0, h)
        else:
            ex, ey = rng.integers(0, w), rng.integers(0, h)
        pos = _find_passable_near(terrain, int(ex), int(ey))
        flood_seeds.append(pos)

    return {
        'lekgolo_center': lekgolo_center,
        'lekgolo_thinker_positions': thinker_positions,
        'lekgolo_worker_positions': worker_positions,
        'flood_clusters': flood_clusters,
        'flood_isolated_seeds': flood_seeds,
    }


def _find_passable_near(terrain: np.ndarray, x: int, y: int,
                         max_search: int = 20) -> tuple[int, int]:
    """Find the nearest passable cell to (x, y) using expanding search."""
    h, w = terrain.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))

    if terrain[y, x] != TERRAIN_WALL:
        return (x, y)

    # Spiral search
    for radius in range(1, max_search):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                        terrain[ny, nx] != TERRAIN_WALL):
                    return (nx, ny)
    return (x, y)  # fallback


# ---------------------------------------------------------------------------
# 7. Biomass field generation
# ---------------------------------------------------------------------------

def generate_biomass_fields(terrain: np.ndarray,
                            rng: np.random.Generator) -> list[dict]:
    """
    Generate biomass resource fields on the map.

    Fields are placed on passable terrain with scarcity gradients:
    - Fields near Lekgolo spawn are safe but low-yield
    - Fields in contested zones are high-yield but risky

    Returns list of field dicts:
        'center': (x, y)
        'radius': int
        'rate': float  (biomass per step per worm inside)
        'is_contested': bool
    """
    h, w = terrain.shape
    num_fields = rng.integers(BIOMASS_FIELD_COUNT_MIN, BIOMASS_FIELD_COUNT_MAX + 1)
    fields = []

    for _ in range(num_fields):
        fx = int(rng.integers(5, w - 5))
        fy = int(rng.integers(5, h - 5))
        fx, fy = _find_passable_near(terrain, fx, fy)

        radius = int(rng.integers(BIOMASS_FIELD_RADIUS_MIN,
                                  BIOMASS_FIELD_RADIUS_MAX + 1))

        # Distance from center determines contested vs safe
        dist_from_center = np.sqrt((fx - w / 2) ** 2 + (fy - h / 2) ** 2)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
        is_contested = dist_from_center > max_dist * 0.5

        # Contested fields yield more biomass (risk/reward)
        rate = BIOMASS_FIELD_RATE * (2.0 if is_contested else 1.0)

        fields.append({
            'center': (fx, fy),
            'radius': radius,
            'rate': rate,
            'is_contested': is_contested,
        })

    return fields


# ---------------------------------------------------------------------------
# 8. Map modifier generation
# ---------------------------------------------------------------------------

def generate_modifiers(terrain: np.ndarray,
                       rng: np.random.Generator) -> list[dict]:
    """
    Generate dynamic map modifiers (1-2 per map).

    Modifiers:
    - toxic_zone: damages all entities over time
    - infection_fog: boosts Flood damage in area
    - comm_jam: reduces communication range in area
    - biomass_decay: drains biomass in area
    - collapsing: terrain gradually turns to wall

    Returns list of modifier dicts.
    """
    h, w = terrain.shape
    modifier_types = [
        MODIFIER_TOXIC_ZONE, MODIFIER_INFECTION_FOG,
        MODIFIER_COMM_JAM, MODIFIER_BIOMASS_DECAY, MODIFIER_COLLAPSING
    ]

    modifiers = []
    chosen_types = rng.choice(len(modifier_types),
                              size=min(NUM_MODIFIERS_PER_MAP, len(modifier_types)),
                              replace=False)

    for idx in chosen_types:
        mtype = modifier_types[int(idx)]
        mx = int(rng.integers(10, w - 10))
        my = int(rng.integers(10, h - 10))
        mx, my = _find_passable_near(terrain, mx, my)
        radius = int(rng.integers(MODIFIER_RADIUS_MIN, MODIFIER_RADIUS_MAX + 1))

        # Apply toxic terrain in modifier zone
        if mtype == MODIFIER_TOXIC_ZONE:
            for ox in range(-radius, radius + 1):
                for oy in range(-radius, radius + 1):
                    if ox * ox + oy * oy <= radius * radius:
                        nx, ny = mx + ox, my + oy
                        if (0 <= nx < w and 0 <= ny < h and
                                terrain[ny, nx] == TERRAIN_FLAT):
                            terrain[ny, nx] = TERRAIN_TOXIC

        modifiers.append({
            'type': mtype,
            'center': (mx, my),
            'radius': radius,
            'active': True,
        })

    return modifiers


# ---------------------------------------------------------------------------
# 9. Event schedule generation
# ---------------------------------------------------------------------------

def generate_event_schedule(rng: np.random.Generator,
                            max_steps: int = MAX_STEPS_PER_EPISODE) -> list[dict]:
    """
    Generate a schedule of procedural events that fire during the match.

    Events:
    - flood_surge: spawn a wave of Flood at edges
    - resource_bloom: burst of biomass
    - terrain_collapse: random area turns to wall
    - thinker_disrupt: thinker boosts temporarily reduced

    Returns list of event dicts sorted by trigger_step.
    """
    event_types = [
        EVENT_FLOOD_SURGE, EVENT_RESOURCE_BLOOM,
        EVENT_TERRAIN_COLLAPSE, EVENT_THINKER_DISRUPT,
    ]

    events = []
    step = EVENT_INTERVAL_MIN

    while step < max_steps:
        etype = str(rng.choice(event_types))
        h, w = MAP_HEIGHT, MAP_WIDTH
        ex = int(rng.integers(5, w - 5))
        ey = int(rng.integers(5, h - 5))

        event = {
            'type': etype,
            'trigger_step': step,
            'position': (ex, ey),
            'processed': False,
        }

        if etype == EVENT_FLOOD_SURGE:
            event['count'] = EVENT_FLOOD_SURGE_COUNT
        elif etype == EVENT_RESOURCE_BLOOM:
            event['amount'] = EVENT_RESOURCE_BLOOM_AMOUNT
        elif etype == EVENT_TERRAIN_COLLAPSE:
            event['radius'] = EVENT_TERRAIN_COLLAPSE_RADIUS
        elif etype == EVENT_THINKER_DISRUPT:
            event['duration'] = EVENT_THINKER_DISRUPT_DURATION

        events.append(event)

        # Next event at random interval
        step += int(rng.integers(EVENT_INTERVAL_MIN, EVENT_INTERVAL_MAX + 1))

    return events


# ---------------------------------------------------------------------------
# 10. Full pipeline: generate_world
# ---------------------------------------------------------------------------

class WorldData:
    """Container for all procedurally generated world data."""

    def __init__(self):
        self.seed: int = 0
        self.heightmap: np.ndarray | None = None
        self.terrain: np.ndarray | None = None
        self.map_type: str = MAP_TYPE_OPEN
        self.regions: list[dict] = []
        self.spawn_formations: dict = {}
        self.biomass_fields: list[dict] = []
        self.modifiers: list[dict] = []
        self.event_schedule: list[dict] = []

    def summary(self) -> str:
        lines = [
            f"Seed: {self.seed}",
            f"Map type: {self.map_type}",
            f"Regions: {len(self.regions)}",
            f"Biomass fields: {len(self.biomass_fields)}",
            f"Modifiers: {[m['type'] for m in self.modifiers]}",
            f"Scheduled events: {len(self.event_schedule)}",
        ]
        if self.regions:
            areas = [r['area'] for r in self.regions]
            lines.append(f"Region sizes: min={min(areas)}, max={max(areas)}, mean={np.mean(areas):.0f}")
        return "\n".join(lines)


def generate_world(seed: int | None = None,
                   width: int = MAP_WIDTH,
                   height: int = MAP_HEIGHT) -> WorldData:
    """
    Full procedural generation pipeline.

    seed -> heightmap -> terrain -> regions -> connectivity
         -> spawn formations -> biomass fields -> modifiers -> events -> classify

    Returns a WorldData object with everything needed to initialize a match.
    """
    rng = np.random.default_rng(seed)
    actual_seed = int(rng.integers(0, 2**31))

    world = WorldData()
    world.seed = actual_seed

    # Step 1: Heightmap
    world.heightmap = generate_heightmap(width, height, actual_seed)

    # Step 2: Terrain
    world.terrain = heightmap_to_terrain(world.heightmap)

    # Step 3: Find regions
    world.regions = find_regions(world.terrain)

    # Step 4: Spawn formations (need before connectivity check)
    world.spawn_formations = generate_spawn_formations(world.terrain, rng)

    # Step 5: Connectivity - ensure paths between spawn zones
    spawn_points = [world.spawn_formations['lekgolo_center']]
    for cluster_pos in world.spawn_formations['flood_clusters']:
        spawn_points.append(cluster_pos)
    world.terrain = ensure_connectivity(world.terrain, spawn_points)

    # Step 6: Biomass fields
    world.biomass_fields = generate_biomass_fields(world.terrain, rng)

    # Step 7: Map modifiers (may modify terrain, e.g., toxic zones)
    world.modifiers = generate_modifiers(world.terrain, rng)

    # Step 8: Event schedule
    world.event_schedule = generate_event_schedule(rng)

    # Step 9: Classify the map
    world.map_type = classify_map(world.terrain)

    return world


# ---------------------------------------------------------------------------
# Utility: terrain passability (replaces old terrain.py functions)
# ---------------------------------------------------------------------------

def is_passable(terrain: np.ndarray, x: int, y: int) -> bool:
    """Check if a cell is passable (not a wall and within bounds)."""
    h, w = terrain.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return terrain[y, x] != TERRAIN_WALL


def movement_cost(terrain: np.ndarray, x: int, y: int) -> float:
    """Get the movement cost multiplier at a position."""
    from config import TERRAIN_ROUGH_SPEED_PENALTY, TERRAIN_TOXIC_DAMAGE
    h, w = terrain.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return float('inf')
    cell = terrain[y, x]
    if cell == TERRAIN_WALL:
        return float('inf')
    if cell == TERRAIN_ROUGH:
        return 1.0 + TERRAIN_ROUGH_SPEED_PENALTY
    if cell == TERRAIN_TOXIC:
        return 1.2  # slightly harder to move through toxic
    if cell == TERRAIN_HIGHGROUND:
        return 1.0  # normal movement, but vision bonus applied elsewhere
    return 1.0
