"""
Observation/sensor system for each worm.

Each worm receives:
- Local terrain
- Nearby worms (positions, types, health, signals)
- Nearby enemies (positions, health)
- Own state (health, energy, attachments, infection status)
- Communication signals from neighbors
"""
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
PER_WORM_DIM = 7   # dx, dy, type, health_frac, signal(4)
PER_ENEMY_DIM = 4  # dx, dy, health_frac, threat_level
OBSERVATION_DIM = OWN_STATE_DIM + MAX_NEARBY_WORMS * PER_WORM_DIM + MAX_NEARBY_ENEMIES * PER_ENEMY_DIM


def build_observation(worm, all_worms: list, flood_list: list,
                      terrain: np.ndarray, thinker_boost: dict) -> np.ndarray:
    """
    Build the observation vector for a single worm.

    Returns a numpy array of shape (OBSERVATION_DIM,)
    """
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)

    # --- Own state ---
    obs[0] = worm.health / worm.max_health if worm.max_health > 0 else 0.0
    obs[1] = worm.energy / 200.0  # normalized by WORM_MAX_ENERGY
    obs[2] = np.sin(worm.orientation)
    obs[3] = np.cos(worm.orientation)
    obs[4] = float(worm.worm_type)
    obs[5] = 1.0 if worm.infected else 0.0
    obs[6] = len(worm.attachments) / 6.0  # normalized by MAX_ATTACHMENTS
    from attachment_system import AttachmentSystem
    # structural strength will be computed externally; placeholder here
    obs[7] = 0.0  # filled in by environment
    obs[8] = min(worm.local_damage_taken / 50.0, 1.0)
    # Own signal
    obs[9:9 + SIGNAL_DIM] = worm.signal
    # Thinker boosts
    obs[13] = thinker_boost.get('attack_accuracy', 0.0)
    obs[14] = thinker_boost.get('move_efficiency', 0.0)
    # Nearby worm count placeholder
    obs[15] = 0.0

    # --- Nearby worms ---
    nearby_worms = []
    for other in all_worms:
        if other.id == worm.id or not other.alive:
            continue
        d = worm.distance_to(other.x, other.y)
        if d <= worm.vision_radius:
            nearby_worms.append((d, other))

    # Sort by distance, take closest
    nearby_worms.sort(key=lambda x: x[0])
    nearby_worms = nearby_worms[:MAX_NEARBY_WORMS]
    obs[15] = min(len(nearby_worms) / MAX_NEARBY_WORMS, 1.0)

    offset = OWN_STATE_DIM
    for i, (d, other) in enumerate(nearby_worms):
        base = offset + i * PER_WORM_DIM
        obs[base] = (other.x - worm.x) / worm.vision_radius  # normalized dx
        obs[base + 1] = (other.y - worm.y) / worm.vision_radius  # normalized dy
        obs[base + 2] = float(other.worm_type)
        obs[base + 3] = other.health / other.max_health if other.max_health > 0 else 0.0
        obs[base + 4:base + 4 + SIGNAL_DIM] = other.signal

    # --- Nearby enemies ---
    nearby_enemies = []
    for enemy in flood_list:
        if not enemy.alive:
            continue
        d = worm.distance_to(enemy.x, enemy.y)
        if d <= worm.vision_radius:
            nearby_enemies.append((d, enemy))

    nearby_enemies.sort(key=lambda x: x[0])
    nearby_enemies = nearby_enemies[:MAX_NEARBY_ENEMIES]

    enemy_offset = offset + MAX_NEARBY_WORMS * PER_WORM_DIM
    for i, (d, enemy) in enumerate(nearby_enemies):
        base = enemy_offset + i * PER_ENEMY_DIM
        obs[base] = (enemy.x - worm.x) / worm.vision_radius
        obs[base + 1] = (enemy.y - worm.y) / worm.vision_radius
        obs[base + 2] = enemy.health / enemy.max_health if enemy.max_health > 0 else 0.0
        obs[base + 3] = min(d / worm.vision_radius, 1.0)  # threat (closer = higher)

    return obs


def get_communication_signals(worm, all_worms: list,
                              attachment_system) -> np.ndarray:
    """
    Aggregate communication signals from neighbors (both attached and in range).

    Returns averaged signal vector from communicating neighbors.
    """
    total_signal = np.zeros(SIGNAL_DIM, dtype=np.float32)
    count = 0

    comm_radius = worm.comm_radius

    # From attached worms (always received)
    attached_ids = worm.attachments
    for other in all_worms:
        if other.id == worm.id or not other.alive:
            continue
        if other.id in attached_ids or worm.distance_to(other.x, other.y) <= comm_radius:
            total_signal += other.signal
            count += 1

    if count > 0:
        total_signal /= count

    return total_signal
