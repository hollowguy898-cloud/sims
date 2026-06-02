"""
Reward system for the Lekgolo colony simulation.

Rewards target OUTCOMES, not behaviors.
We never reward: "protect thinkers", "build walls", "form brains",
"swarm thinkers", "probe weak points".
Those should EMERGE from the reward structure.

Colony rewards:
  + survive (per timestep)
  + maintain thinkers (per thinker alive per timestep)
  + maintain connectivity (per connected worker per thinker)
  + destroy Flood (per kill)
  + grow biomass
  - thinker death
  - worker infection
  - thinker infection
  - colony fragmentation

Flood rewards (outcome-based, asymmetric):
  + infect worker/thinker (penetration > killing)
  + biomass growth (replication economy)
  + colony fragmentation (disruption)
  + geographic spread (control territory)
  + survive (per timestep, low - Flood are expendable)
  - Flood deaths (but penalty is small - they're cheap)
"""
import numpy as np
from config import (
    REWARD_SURVIVAL_PER_STEP,
    REWARD_FLOOD_KILL,
    REWARD_WORKER_INFECTED,
    REWARD_THINKER_INFECTED,
    REWARD_THINKER_ALIVE_PER_STEP,
    REWARD_CONNECTED_WORKERS_PER_THINKER,
    REWARD_DAMAGE_BLOCKED,
    REWARD_TERRITORY_PER_CELL,
    REWARD_BIOMASS_CHANGE_SCALE,
    REWARD_THINKER_DEATH,
    REWARD_COLONY_FRAGMENTATION,
    FLOOD_REWARD_INFECT_WORKER,
    FLOOD_REWARD_INFECT_THINKER,
    FLOOD_REWARD_COLONY_FRAGMENT,
    FLOOD_REWARD_DEATH,
    FLOOD_REWARD_BIOMASS_GROWTH,
    FLOOD_REWARD_SURVIVAL_PER_STEP,
    FLOOD_REWARD_SPREAD_BONUS,
    NUM_WORMS_INITIAL,
    NUM_THINKERS_INITIAL,
)


class ColonyRewardTracker:
    """Tracks and computes rewards for the Lekgolo colony."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.prev_biomass = 0.0
        self.prev_fragment_count = 1
        self.reward_breakdown = {
            'survival': 0.0,
            'flood_kills': 0.0,
            'worker_infected': 0.0,
            'thinker_infected': 0.0,
            'thinker_alive': 0.0,
            'connectivity': 0.0,
            'damage_blocked': 0.0,
            'territory': 0.0,
            'biomass_change': 0.0,
            'thinker_death': 0.0,
            'fragmentation': 0.0,
        }

    def compute_step_reward(self, worms: list, flood_list: list,
                            attachment_system, terrain: np.ndarray,
                            current_biomass: float) -> float:
        """Compute the colony-level reward for one timestep."""
        reward = 0.0
        alive_worms = [w for w in worms if w.alive]
        alive_workers = [w for w in alive_worms if w.worm_type == 0]
        alive_thinkers = [w for w in alive_worms if w.worm_type == 1]

        # --- Survival reward ---
        survival_r = len(alive_worms) * REWARD_SURVIVAL_PER_STEP
        reward += survival_r
        self.reward_breakdown['survival'] += survival_r

        # --- Flood kills ---
        total_kills = sum(w.flood_kills_this_step for w in alive_worms)
        kills_r = total_kills * REWARD_FLOOD_KILL
        reward += kills_r
        self.reward_breakdown['flood_kills'] += kills_r

        # --- Infection penalties ---
        for w in worms:
            if w.infected and w.local_damage_taken > 0 and w.alive:
                if w.worm_type == 0:
                    inf_r = REWARD_WORKER_INFECTED
                else:
                    inf_r = REWARD_THINKER_INFECTED
                reward += inf_r
                if w.worm_type == 0:
                    self.reward_breakdown['worker_infected'] += inf_r
                else:
                    self.reward_breakdown['thinker_infected'] += inf_r

        # --- Thinker alive bonus ---
        thinker_r = len(alive_thinkers) * REWARD_THINKER_ALIVE_PER_STEP
        reward += thinker_r
        self.reward_breakdown['thinker_alive'] += thinker_r

        # --- Connectivity reward: thinkers connected to workers ---
        alive_worm_ids = {w.id for w in alive_worms}
        connectivity_r = 0.0
        for thinker in alive_thinkers:
            # Use adjacency index for O(degree) lookup instead of O(E)
            connected = attachment_system.adjacency.get(thinker.id, set())
            connected_alive = [c for c in connected if c in alive_worm_ids]
            connectivity_r += len(connected_alive) * REWARD_CONNECTED_WORKERS_PER_THINKER
        reward += connectivity_r
        self.reward_breakdown['connectivity'] += connectivity_r

        # --- Damage blocked ---
        blocked_r = sum(w.damage_blocked_this_step for w in alive_worms) * REWARD_DAMAGE_BLOCKED
        reward += blocked_r
        self.reward_breakdown['damage_blocked'] += blocked_r

        # --- Territory control ---
        if alive_worms:
            xs = [w.x for w in alive_worms]
            ys = [w.y for w in alive_worms]
            territory = max(0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            territory_r = territory * REWARD_TERRITORY_PER_CELL
            reward += territory_r
            self.reward_breakdown['territory'] += territory_r

        # --- Biomass change ---
        biomass_change = current_biomass - self.prev_biomass
        biomass_r = biomass_change * REWARD_BIOMASS_CHANGE_SCALE
        reward += biomass_r
        self.reward_breakdown['biomass_change'] += biomass_r
        self.prev_biomass = current_biomass

        # --- Thinker death penalty ---
        for w in worms:
            if not w.alive and w.worm_type == 1 and w.local_damage_taken > 0:
                if w.health == 0:
                    death_r = REWARD_THINKER_DEATH
                    reward += death_r
                    self.reward_breakdown['thinker_death'] += death_r

        # --- Colony fragmentation ---
        current_fragments = attachment_system.count_fragments(alive_worm_ids)
        if current_fragments > self.prev_fragment_count:
            new_fragments = current_fragments - self.prev_fragment_count
            frag_r = new_fragments * REWARD_COLONY_FRAGMENTATION
            reward += frag_r
            self.reward_breakdown['fragmentation'] += frag_r
        self.prev_fragment_count = current_fragments

        self.total_reward += reward
        return reward


class FloodRewardTracker:
    """
    Tracks and computes rewards for the Flood.

    Flood rewards are deliberately asymmetric:
    - Infection is rewarded far more than killing (conversion economy)
    - Spread is rewarded (territory control)
    - Death penalty is small (Flood are expendable)
    - Colony fragmentation is a major goal (disruption)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward = 0.0
        self.reward_breakdown = {
            'infect_worker': 0.0,
            'infect_thinker': 0.0,
            'colony_fragment': 0.0,
            'death': 0.0,
            'biomass_growth': 0.0,
            'survival': 0.0,
            'spread': 0.0,
        }

    def compute_step_reward(self, flood_list: list, prev_count: int,
                            colony_fragmented: bool,
                            biomass_gained: float) -> float:
        """Compute Flood-level reward for one timestep."""
        reward = 0.0
        alive_flood = [f for f in flood_list if f.alive]

        # Survival (very small - Flood are expendable)
        survival_r = len(alive_flood) * FLOOD_REWARD_SURVIVAL_PER_STEP
        reward += survival_r
        self.reward_breakdown['survival'] += survival_r

        # Flood deaths (small penalty)
        current_alive = len(alive_flood)
        deaths = max(0, prev_count - current_alive)
        death_r = deaths * FLOOD_REWARD_DEATH
        reward += death_r
        self.reward_breakdown['death'] += death_r

        # Colony fragmentation (major goal)
        if colony_fragmented:
            frag_r = FLOOD_REWARD_COLONY_FRAGMENT
            reward += frag_r
            self.reward_breakdown['colony_fragment'] += frag_r

        # Biomass growth
        bio_r = biomass_gained * FLOOD_REWARD_BIOMASS_GROWTH
        reward += bio_r
        self.reward_breakdown['biomass_growth'] += bio_r

        # Geographic spread bonus
        if alive_flood:
            xs = [f.x for f in alive_flood]
            ys = [f.y for f in alive_flood]
            spread_area = max(0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
            spread_r = spread_area * FLOOD_REWARD_SPREAD_BONUS
            reward += spread_r
            self.reward_breakdown['spread'] += spread_r

        self.total_reward += reward
        return reward

    def compute_infection_reward(self, worm_type: int) -> float:
        """Immediate reward when a Flood infects a worm."""
        if worm_type == 1:  # Thinker
            r = FLOOD_REWARD_INFECT_THINKER
            self.reward_breakdown['infect_thinker'] += r
        else:
            r = FLOOD_REWARD_INFECT_WORKER
            self.reward_breakdown['infect_worker'] += r
        self.total_reward += r
        return r
