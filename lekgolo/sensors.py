"""
Observation/sensor system for each worm.

Each worm receives:
- Local terrain
- Nearby worms (positions, types, health, signals)
- Nearby enemies (positions, health)
- Own state (health, energy, attachments, infection status)
- Communication signals from neighbors

OPTIMIZED: build_observation uses pre-computed nearby_worms/nearby_enemies
from the spatial grid, avoiding redundant O(n²) scans.
"""
import math
import numpy as np
from config import (
    SIGNAL_DIM, WORM_VISION_RADIUS_WORKER, WORM_VISION_RADIUS_THINKER,
    MAP_WIDTH, MAP_HEIGHT, TERRAIN_FLAT, TERRAIN_ROUGH, TERRAIN_WALL
)


# Maximum number of nearby entities we encode in the observation
MAX_NEARBY_WORMS = 8
MAX_NEARBY_ENEMIES = 8

# Observation vector layout:
# [0:4]   - own state: health_frac, energy_frac, orientation_sin, orientation_cos
# [4]     - own type (0=worker, 1=thinker)
# [5]     - is_infected
# [6]     - num_attachments / MAX_ATTACHMENTS
# [7]     - structural_strength_fraction
# [8]     - local_damage_taken (normalized)
# [9:13]  - own signal (SIGNAL_DIM=4)
# [13]    - thinker_boost_attack_accuracy
# [14]    - thinker_boost_move_efficiency
# [15]    - nearby_worm_count (normalized)
# Then per nearby worm (MAX_NEARBY_WORMS * 7):
#   [dx, dy, type, health_frac, signal(4)]
# Then per nearby enemy (MAX_NEARBY_ENEMIES * 4):
#   [dx, dy, health_frac, is_thinker_target]

OWN_STATE_DIM = 16
PER_WORM_DIM = 7
PER_ENEMY_DIM = 4
OBSERVATION_DIM = OWN_STATE_DIM + MAX_NEARBY_WORMS * PER_WORM_DIM + MAX_NEARBY_ENEMIES * PER_ENEMY_DIM

# Pre-compute offsets
_NEARBY_WORM_OFFSET = OWN_STATE_DIM
_NEARBY_ENEMY_OFFSET = _NEARBY_WORM_OFFSET + MAX_NEARBY_WORMS * PER_WORM_DIM


def build_observation(worm, thinker_boost: dict,
                      obs_out: np.ndarray | None = None) -> np.ndarray:
    """
    Build the observation vector for a single worm.

    Uses worm.nearby_worms and worm.nearby_enemies (pre-computed
    by the spatial grid in _build_all_observations), avoiding
    redundant O(n²) distance scans.

    Optionally writes into a pre-allocated buffer (obs_out).
    """
    if obs_out is None:
        obs_out = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    else:
        obs_out[:] = 0.0

    # --- Own state ---
    obs_out[0] = worm.health / worm.max_health if worm.max_health > 0 else 0.0
    obs_out[1] = worm.energy / 200.0
    obs_out[2] = math.sin(worm.orientation)
    obs_out[3] = math.cos(worm.orientation)
    obs_out[4] = float(worm.worm_type)
    obs_out[5] = 1.0 if worm.infected else 0.0
    obs_out[6] = len(worm.attachments) / 6.0
    # obs_out[7] filled in by environment (structural strength)
    obs_out[8] = min(worm.local_damage_taken / 50.0, 1.0)
    # Own signal
    for i in range(SIGNAL_DIM):
        obs_out[9 + i] = worm.signal[i]
    # Thinker boosts
    obs_out[13] = thinker_boost.get('attack_accuracy', 0.0)
    obs_out[14] = thinker_boost.get('move_efficiency', 0.0)

    # --- Nearby worms (from pre-computed list) ---
    nearby = worm.nearby_worms
    # Sort by distance, take closest MAX_NEARBY_WORMS
    if len(nearby) > MAX_NEARBY_WORMS:
        # Partial sort - only need top MAX_NEARBY_WORMS
        nearby_sorted = sorted(nearby,
                               key=lambda o: worm.distance_to_sq(o.x, o.y))
        nearby_sorted = nearby_sorted[:MAX_NEARBY_WORMS]
    else:
        nearby_sorted = nearby

    count = len(nearby_sorted)
    obs_out[15] = min(count / MAX_NEARBY_WORMS, 1.0)
    inv_vision = 1.0 / worm.vision_radius

    for i, other in enumerate(nearby_sorted):
        base = _NEARBY_WORM_OFFSET + i * PER_WORM_DIM
        obs_out[base] = (other.x - worm.x) * inv_vision
        obs_out[base + 1] = (other.y - worm.y) * inv_vision
        obs_out[base + 2] = float(other.worm_type)
        obs_out[base + 3] = other.health / other.max_health if other.max_health > 0 else 0.0
        for j in range(SIGNAL_DIM):
            obs_out[base + 4 + j] = other.signal[j]

    # --- Nearby enemies (from pre-computed list) ---
    nearby_e = worm.nearby_enemies
    if len(nearby_e) > MAX_NEARBY_ENEMIES:
        nearby_e_sorted = sorted(nearby_e,
                                 key=lambda e: worm.distance_to_sq(e.x, e.y))
        nearby_e_sorted = nearby_e_sorted[:MAX_NEARBY_ENEMIES]
    else:
        nearby_e_sorted = nearby_e

    for i, enemy in enumerate(nearby_e_sorted):
        base = _NEARBY_ENEMY_OFFSET + i * PER_ENEMY_DIM
        d = worm.distance_to(enemy.x, enemy.y)
        obs_out[base] = (enemy.x - worm.x) * inv_vision
        obs_out[base + 1] = (enemy.y - worm.y) * inv_vision
        obs_out[base + 2] = enemy.health / enemy.max_health if enemy.max_health > 0 else 0.0
        obs_out[base + 3] = min(d * inv_vision, 1.0)

    return obs_out


def get_communication_signals(worm, alive_worms_by_id: dict,
                              attachment_system) -> np.ndarray:
    """
    Aggregate communication signals from neighbors (both attached and in range).

    Uses adjacency index for O(degree) attached-worm lookup
    and spatial grid for O(1) range queries.
    """
    total_signal = [0.0] * SIGNAL_DIM
    count = 0

    comm_radius = worm.comm_radius
    comm_radius_sq = worm.comm_radius_sq

    # From attached worms (always received) — O(degree) via adjacency index
    attached_ids = worm.attachments
    for nid in attached_ids:
        other = alive_worms_by_id.get(nid)
        if other is not None and other.alive:
            for j in range(SIGNAL_DIM):
                total_signal[j] += other.signal[j]
            count += 1

    # From worms in comm range (using pre-computed nearby list)
    for other in worm.nearby_worms:
        if other.id == worm.id or not other.alive:
            continue
        if other.id in attached_ids:
            continue  # already counted above
        d_sq = worm.distance_to_sq(other.x, other.y)
        if d_sq <= comm_radius_sq:
            for j in range(SIGNAL_DIM):
                total_signal[j] += other.signal[j]
            count += 1

    if count > 0:
        inv_count = 1.0 / count
        return np.array([s * inv_count for s in total_signal], dtype=np.float32)

    return np.zeros(SIGNAL_DIM, dtype=np.float32)
