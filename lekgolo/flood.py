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


def build_flood_observation(flood, nearby_worms: list, nearby_flood: list,
                            terrain: np.ndarray, biomass_fields: list,
                            worms_by_id: dict,
                            attachment_system=None) -> np.ndarray:
    """
    Build the observation vector for a single Flood agent.

    Includes vulnerability signals: isolated thinkers, thin walls,
    colony density.
    """
    obs = np.zeros(FLOOD_OBSERVATION_DIM, dtype=np.float32)

    # --- Own state ---
    obs[0] = flood.health / flood.max_health if flood.max_health > 0 else 0.0
    obs[1] = np.sin(flood.orientation)
    obs[2] = np.cos(flood.orientation)
    obs[3] = flood.biomass / (FLOOD_REPRODUCE_BIOMASS_COST * 3)  # normalized
    obs[4] = len(nearby_worms) / 20.0  # nearby infection count proxy

    # Own signal
    obs[5:5 + FLOOD_SIGNAL_DIM] = flood.signal

    # Nearby densities
    obs[9] = min(len(nearby_worms) / 20.0, 1.0)
    obs[10] = min(len(nearby_flood) / 15.0, 1.0)

    # Local biomass availability (check if in a biomass field)
    local_biomass = 0.0
    for field in biomass_fields:
        fx, fy = field['center']
        d = flood.distance_to(fx, fy)
        if d <= field['radius']:
            local_biomass = field['rate'] / 5.0  # normalize
            break
    obs[11] = min(local_biomass, 1.0)

    # Vulnerability signal: is there an isolated thinker nearby?
    vulnerability = 0.0
    for worm in nearby_worms:
        if worm.worm_type == 1 and worm.alive:  # Thinker
            # Check if isolated (few attachments)
            if len(worm.attachments) <= 1:
                vulnerability = 1.0
                break
    obs[12] = vulnerability

    # --- Nearby worms (Lekgolo) ---
    worm_data = []
    for worm in nearby_worms:
        if not worm.alive:
            continue
        d = flood.distance_to(worm.x, worm.y)
        if d <= FLOOD_VISION:
            is_isolated = len(worm.attachments) <= 1
            worm_data.append((d, worm, is_isolated))

    worm_data.sort(key=lambda x: x[0])
    worm_data = worm_data[:FLOOD_MAX_NEARBY_WORMS]

    offset = FLOOD_OWN_STATE_DIM
    for i, (d, worm, is_isolated) in enumerate(worm_data):
        base = offset + i * FLOOD_PER_WORM_DIM
        obs[base] = (worm.x - flood.x) / FLOOD_VISION
        obs[base + 1] = (worm.y - flood.y) / FLOOD_VISION
        obs[base + 2] = 1.0 if worm.worm_type == 1 else 0.0  # is_thinker
        obs[base + 3] = worm.health / worm.max_health if worm.max_health > 0 else 0.0
        obs[base + 4] = 1.0 if is_isolated else 0.0

    # --- Nearby Flood ---
    flood_data = []
    for other in nearby_flood:
        if not other.alive:
            continue
        d = flood.distance_to(other.x, other.y)
        if d <= FLOOD_VISION:
            flood_data.append((d, other))

    flood_data.sort(key=lambda x: x[0])
    flood_data = flood_data[:FLOOD_MAX_NEARBY_FLOOD]

    flood_offset = offset + FLOOD_MAX_NEARBY_WORMS * FLOOD_PER_WORM_DIM
    for i, (d, other) in enumerate(flood_data):
        base = flood_offset + i * FLOOD_PER_FLOOD_DIM
        obs[base] = (other.x - flood.x) / FLOOD_VISION
        obs[base + 1] = (other.y - flood.y) / FLOOD_VISION
        obs[base + 2] = other.health / other.max_health if other.max_health > 0 else 0.0

    return obs


# ---------------------------------------------------------------------------
# Flood actions
# ---------------------------------------------------------------------------

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
    params = action_vector[1:1 + FLOOD_ACTION_PARAM_DIM].copy()
    return {'type': action_type, 'params': params}


# ---------------------------------------------------------------------------
# Flood agent class
# ---------------------------------------------------------------------------

class FloodOrganism:
    """
    A single Flood organism - an RL agent.

    Flood are designed as a rival lifeform, not a scripted villain:
    - High birth rate, low individual value
    - Reward scales with spread and disruption
    - Simpler neural architecture than Lekgolo
    """

    _next_id = 0

    def __init__(self, x: float, y: float):
        self.id = FloodOrganism._next_id
        FloodOrganism._next_id += 1

        self.x = x
        self.y = y
        self.health = FLOOD_HEALTH
        self.max_health = FLOOD_HEALTH
        self.strength = FLOOD_STRENGTH
        self.alive = True
        self.orientation = np.random.uniform(0, 2 * np.pi)
        self.reproduce_timer = FLOOD_REPRODUCE_INTERVAL
        self.biomass = FLOOD_REPRODUCE_BIOMASS_COST  # starts ready to reproduce

        # Communication signal - meaning is learned, not defined
        self.signal = np.zeros(FLOOD_SIGNAL_DIM, dtype=np.float32)

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

    def distance_to(self, other_x: float, other_y: float) -> float:
        return np.sqrt((self.x - other_x) ** 2 + (self.y - other_y) ** 2)

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
