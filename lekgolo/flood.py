"""
Flood organism - an RL agent, not a scripted adversary.

Each Flood unit is an agent with:
- State: position, health, infection/biomass level, signal
- Observations: nearby worms (Lekgolo or Flood), local biomass,
  colony density, vulnerability signals
- Actions: move, attack, infect, split/spawn, signal

Flood are deliberately simpler than Lekgolo:
  - Smaller network (32 hidden vs 16/128 for workers/thinkers)
  - Fewer actions (5 vs 6)
  - Focus on replication + disruption, not structure

This asymmetry is key: Lekgolo = structure + coordination,
Flood = replication + aggression. Different "physics of intelligence."
"""
import math
import numpy as np
from config import (
    FLOOD_HEALTH, FLOOD_STRENGTH, FLOOD_SPEED, FLOOD_ATTACK_DAMAGE,
    FLOOD_INFECTION_DAMAGE, FLOOD_ATTACK_RANGE, FLOOD_VISION,
    FLOOD_REPRODUCE_INTERVAL, FLOOD_REPRODUCE_BIOMASS_COST,
    FLOOD_SIGNAL_DIM, FLOOD_COMM_RADIUS, MAP_WIDTH, MAP_HEIGHT,
    FLOOD_NUM_DISCRETE_ACTIONS, FLOOD_ACTION_PARAM_DIM,
    FLOOD_MAX_NEARBY_WORMS, FLOOD_MAX_NEARBY_FLOOD,
)


# ---------------------------------------------------------------------------
# Flood observation vector
# ---------------------------------------------------------------------------

# Layout:
# [0:5]   own state: health_frac, orientation_sin, orientation_cos,
#          biomass_frac, infection_count (nearby)
# [5:9]   own signal (FLOOD_SIGNAL_DIM=4)
# [9]     nearby_worm_density (normalized)
# [10]    nearby_flood_density (normalized)
# [11]    local_biomass_availability (normalized)
# [12]    vulnerability_signal (is_thinker_nearby_and_isolated)
# then per nearby worm (FLOOD_MAX_NEARBY_WORMS * 5):
#   [dx, dy, is_thinker, health_frac, is_isolated]
# then per nearby flood (FLOOD_MAX_NEARBY_FLOOD * 3):
#   [dx, dy, health_frac]

FLOOD_OWN_STATE_DIM = 13
FLOOD_PER_WORM_DIM = 5
FLOOD_PER_FLOOD_DIM = 3
FLOOD_OBSERVATION_DIM = (FLOOD_OWN_STATE_DIM +
                         FLOOD_MAX_NEARBY_WORMS * FLOOD_PER_WORM_DIM +
                         FLOOD_MAX_NEARBY_FLOOD * FLOOD_PER_FLOOD_DIM)

# Pre-compute offsets
_FLOOD_WORM_OFFSET = FLOOD_OWN_STATE_DIM
_FLOOD_FLOOD_OFFSET = _FLOOD_WORM_OFFSET + FLOOD_MAX_NEARBY_WORMS * FLOOD_PER_WORM_DIM

# Squared vision for fast comparisons
_FLOOD_VISION_SQ = FLOOD_VISION * FLOOD_VISION
_FLOOD_ATTACK_RANGE_SQ = FLOOD_ATTACK_RANGE * FLOOD_ATTACK_RANGE


def build_flood_observation(flood, nearby_worms: list, nearby_flood: list,
                            terrain: np.ndarray, biomass_fields: list,
                            worms_by_id: dict,
                            attachment_system=None,
                            obs_out: np.ndarray | None = None) -> np.ndarray:
    """
    Build the observation vector for a single Flood agent.

    Uses squared distance to avoid sqrt in hot loops.
    Optionally writes into a pre-allocated buffer (obs_out).
    """
    if obs_out is None:
        obs_out = np.zeros(FLOOD_OBSERVATION_DIM, dtype=np.float32)
    else:
        obs_out[:] = 0.0

    # --- Own state ---
    obs_out[0] = flood.health / flood.max_health if flood.max_health > 0 else 0.0
    obs_out[1] = math.sin(flood.orientation)
    obs_out[2] = math.cos(flood.orientation)
    obs_out[3] = flood.biomass / (FLOOD_REPRODUCE_BIOMASS_COST * 3)
    obs_out[4] = len(nearby_worms) / 20.0

    # Own signal
    for i in range(FLOOD_SIGNAL_DIM):
        obs_out[5 + i] = flood.signal[i]

    # Nearby densities
    obs_out[9] = min(len(nearby_worms) / 20.0, 1.0)
    obs_out[10] = min(len(nearby_flood) / 15.0, 1.0)

    # Local biomass availability
    local_biomass = 0.0
    for field in biomass_fields:
        fx, fy = field['center']
        d = flood.distance_to(fx, fy)
        if d <= field['radius']:
            local_biomass = field['rate'] / 5.0
            break
    obs_out[11] = min(local_biomass, 1.0)

    # Vulnerability signal: is there an isolated thinker nearby?
    vulnerability = 0.0
    for worm in nearby_worms:
        if worm.worm_type == 1 and worm.alive:
            if len(worm.attachments) <= 1:
                vulnerability = 1.0
                break
    obs_out[12] = vulnerability

    # --- Nearby worms (Lekgolo) ---
    # nearby_worms already filtered by distance in _build_all_flood_observations
    # Sort by distance (already pre-sorted in the spatial grid query)
    worm_list = [(flood.distance_to(w.x, w.y), w) for w in nearby_worms
                 if w.alive and flood.distance_to_sq(w.x, w.y) <= _FLOOD_VISION_SQ]
    worm_list.sort(key=lambda x: x[0])
    worm_list = worm_list[:FLOOD_MAX_NEARBY_WORMS]

    for i, (d, worm) in enumerate(worm_list):
        base = _FLOOD_WORM_OFFSET + i * FLOOD_PER_WORM_DIM
        obs_out[base] = (worm.x - flood.x) / FLOOD_VISION
        obs_out[base + 1] = (worm.y - flood.y) / FLOOD_VISION
        obs_out[base + 2] = 1.0 if worm.worm_type == 1 else 0.0
        obs_out[base + 3] = worm.health / worm.max_health if worm.max_health > 0 else 0.0
        obs_out[base + 4] = 1.0 if len(worm.attachments) <= 1 else 0.0

    # --- Nearby Flood ---
    flood_list = [(flood.distance_to(o.x, o.y), o) for o in nearby_flood
                  if o.alive and o.id != flood.id and
                  flood.distance_to_sq(o.x, o.y) <= _FLOOD_VISION_SQ]
    flood_list.sort(key=lambda x: x[0])
    flood_list = flood_list[:FLOOD_MAX_NEARBY_FLOOD]

    for i, (d, other) in enumerate(flood_list):
        base = _FLOOD_FLOOD_OFFSET + i * FLOOD_PER_FLOOD_DIM
        obs_out[base] = (other.x - flood.x) / FLOOD_VISION
        obs_out[base + 1] = (other.y - flood.y) / FLOOD_VISION
        obs_out[base + 2] = other.health / other.max_health if other.max_health > 0 else 0.0

    return obs_out


def decode_flood_action(action_vector: np.ndarray) -> dict:
    """
    Decode a Flood action vector.

    Actions:
      0: Move (dx, dy)
      1: Attack nearest worm
      2: Infect (1:1 conversion attempt)
      3: Split/Spawn (if biomass allows)
      4: Signal
    """
    action_type = int(np.clip(np.round(action_vector[0]),
                              0, FLOOD_NUM_DISCRETE_ACTIONS - 1))
    params = action_vector[1:1 + FLOOD_ACTION_PARAM_DIM]
    return {'type': action_type, 'params': params}


class FloodOrganism:
    """
    A single Flood organism - an RL agent.

    Flood are designed as a rival lifeform, not a scripted villain:
    - High birth rate, low individual value
    - Reward scales with spread and disruption
    - Simpler neural architecture than Lekgolo
    """

    _next_id = 0

    def __init__(self, x: float, y: float, rng=None):
        self.id = FloodOrganism._next_id
        FloodOrganism._next_id += 1

        self.x = x
        self.y = y
        self.health = FLOOD_HEALTH
        self.max_health = FLOOD_HEALTH
        self.strength = FLOOD_STRENGTH
        self.alive = True
        if rng is not None:
            self.orientation = rng.uniform(0, 2 * math.pi)
        else:
            self.orientation = np.random.uniform(0, 2 * math.pi)
        self.reproduce_timer = FLOOD_REPRODUCE_INTERVAL
        self.biomass = FLOOD_REPRODUCE_BIOMASS_COST

        # Communication signal - meaning is learned, not defined
        self.signal = [0.0] * FLOOD_SIGNAL_DIM  # list for speed

        # Sensor caches (updated each timestep)
        self.nearby_worms: list = []
        self.nearby_flood: list = []

        # Tracking for rewards
        self.infections_this_step: int = 0
        self.kills_this_step: int = 0
        self.damage_dealt_this_step: float = 0.0

    def take_damage(self, amount: float) -> float:
        if not self.alive:
            return 0.0
        actual = min(amount, self.health)
        self.health -= actual
        if self.health <= 0:
            self.health = 0
            self.alive = False
        return actual

    def distance_to_sq(self, other_x: float, other_y: float) -> float:
        """Squared distance — avoids sqrt."""
        dx = self.x - other_x
        dy = self.y - other_y
        return dx * dx + dy * dy

    def distance_to(self, other_x: float, other_y: float) -> float:
        """Exact distance."""
        dx = self.x - other_x
        dy = self.y - other_y
        return math.sqrt(dx * dx + dy * dy)

    def tick_timers(self):
        if self.alive:
            self.reproduce_timer = max(0, self.reproduce_timer - 1)

    def can_reproduce(self) -> bool:
        return (self.alive and
                self.reproduce_timer <= 0 and
                self.biomass >= FLOOD_REPRODUCE_BIOMASS_COST)

    def reset_step_trackers(self):
        self.infections_this_step = 0
        self.kills_this_step = 0
        self.damage_dealt_this_step = 0.0
