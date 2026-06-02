"""
Main Lekgolo Colony simulation environment.

Uses the full procedural generation pipeline and treats both sides
as RL agents:
  Lekgolo colony: distributed organism (structure + coordination + thinker protection)
  Flood: self-replicating parasite colony (replication + disruption + spread)

Both sides learn. Both sides adapt. Neither is scripted.

OPTIMIZED:
- SpatialGrid for O(1) neighbor lookups instead of O(n²)
- Single observation build per step (cached for return)
- Batch thinker boost computation (once per step)
- Squared distance comparisons (no sqrt in hot paths)
- Pre-allocated observation buffers
- Fixed biomass decay modifier bug
"""
import math
import numpy as np
import os
import torch
from collections import defaultdict

from config import (
    MAP_WIDTH, MAP_HEIGHT, MAX_STEPS_PER_EPISODE,
    NUM_WORMS_INITIAL, NUM_THINKERS_INITIAL, NUM_FLOOD_INITIAL,
    FLOOD_SPAWN_RATE, MAX_FLOOD_COUNT,
    WORM_MAX_ENERGY, WORM_ENERGY_REGEN, WORM_ATTACK_RANGE,
    WORM_ATTACK_DAMAGE_WORKER, WORM_ATTACK_DAMAGE_THINKER,
    WORM_VISION_RADIUS_THINKER,
    SIGNAL_DIM, BIOMASS_COST_WORKER, BIOMASS_COST_THINKER,
    BIOMASS_PER_FLOOD_KILL, BIOMASS_PER_TIMESTEP, INITIAL_BIOMASS,
    FLOOD_HEALTH, FLOOD_ATTACK_DAMAGE, FLOOD_INFECTION_DAMAGE,
    FLOOD_ATTACK_RANGE, FLOOD_SPEED, FLOOD_VISION,
    FLOOD_REPRODUCE_INTERVAL, FLOOD_REPRODUCE_BIOMASS_COST,
    FLOOD_SIGNAL_DIM, FLOOD_COMM_RADIUS,
    ATTACHMENT_MAX_DISTANCE, THINKER_BOOST_RADIUS,
    PPO_ROLLOUT_LENGTH,
    TERRAIN_FLAT, TERRAIN_ROUGH, TERRAIN_WALL, TERRAIN_TOXIC, TERRAIN_HIGHGROUND,
    TERRAIN_TOXIC_DAMAGE, TERRAIN_HIGHGROUND_VISION_BONUS,
    MODIFIER_INFECTION_FOG, MODIFIER_COMM_JAM, MODIFIER_BIOMASS_DECAY,
    MODIFIER_COLLAPSING,
    MODIFIER_INFECTION_FOG_FLOOD_BOOST, MODIFIER_COMM_JAM_RANGE_PENALTY,
    MODIFIER_BIOMASS_DECAY_RATE,
    EVENT_FLOOD_SURGE, EVENT_RESOURCE_BLOOM, EVENT_TERRAIN_COLLAPSE,
    EVENT_THINKER_DISRUPT,
    EVENT_THINKER_DISRUPT_BOOST_PENALTY,
    FLOOD_SPAWN_CLUSTER_COUNT, FLOOD_SPAWN_CLUSTER_SIZE, FLOOD_SPAWN_ISOLATED_SEEDS,
    setup_matplotlib_fonts, DEFAULT_FRAME_DIR,
)
from world_gen import generate_world, is_passable, movement_cost, WorldData
from worm import Worm, WormType
from flood import (FloodOrganism, build_flood_observation,
                   decode_flood_action, FLOOD_OBSERVATION_DIM)
from attachment_system import (AttachmentSystem, compute_thinker_boost,
                               compute_all_thinker_boosts)
from sensors import build_observation, get_communication_signals, OBSERVATION_DIM
from actions import decode_action, process_action, ACTION_VECTOR_DIM
from rewards import ColonyRewardTracker, FloodRewardTracker
from network import ColonySharedPolicy, FloodPolicy
from ppo_trainer import PPOTrainer
from spatial_grid import SpatialGrid

# Pre-compute squared distances
_ATTACH_MAX_DIST_SQ = ATTACHMENT_MAX_DISTANCE * ATTACHMENT_MAX_DISTANCE
_ATTACK_RANGE_SQ = WORM_ATTACK_RANGE * WORM_ATTACK_RANGE
_FLOOD_ATTACK_RANGE_SQ = FLOOD_ATTACK_RANGE * FLOOD_ATTACK_RANGE


class LekgoloEnvironment:
    """
    The main simulation environment.

    Manages the world state and provides step/reset interface
    compatible with RL training for both Lekgolo and Flood.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT

        # World state
        self.world: WorldData | None = None
        self.terrain = np.zeros((self.height, self.width), dtype=np.int8)
        self.worms: list[Worm] = []
        self.flood_list: list[FloodOrganism] = []
        self.worms_by_id: dict[int, Worm] = {}
        self.flood_by_id: dict[int, FloodOrganism] = {}
        self.attachment_system = AttachmentSystem()

        # Spatial grids for O(1) neighbor lookups
        max_vision = max(WORM_VISION_RADIUS_THINKER, FLOOD_VISION,
                         THINKER_BOOST_RADIUS, ATTACHMENT_MAX_DISTANCE) + 1
        self.worm_grid = SpatialGrid(cell_size=max_vision)
        self.flood_grid = SpatialGrid(cell_size=FLOOD_VISION + 1)

        # Colony state
        self.biomass = INITIAL_BIOMASS
        self.timestep = 0
        self.episode_reward = 0.0
        self.flood_episode_reward = 0.0

        # Event/modifier state
        self.thinker_disrupt_timer = 0
        self.active_events: list[dict] = []

        # Reward trackers
        self.colony_reward_tracker = ColonyRewardTracker()
        self.flood_reward_tracker = FloodRewardTracker()

        # RL components
        self.device = 'cpu'
        self.lekgolo_policy = ColonySharedPolicy(OBSERVATION_DIM).to(self.device)
        self.flood_policy = FloodPolicy(FLOOD_OBSERVATION_DIM).to(self.device)
        self.lekgolo_trainer = PPOTrainer(self.lekgolo_policy, self.device)
        self.flood_trainer = PPOTrainer(self.flood_policy, self.device)

        # Pre-allocated observation buffers (resized as needed)
        self._worm_obs_buffer: dict[int, np.ndarray] = {}
        self._flood_obs_buffer: dict[int, np.ndarray] = {}

        # Cached per-step computations
        self._cached_lekgolo_obs: dict | None = None
        self._cached_flood_obs: dict | None = None
        self._cached_boosts: dict | None = None
        self._alive_worms: list[Worm] = []
        self._alive_flood: list[FloodOrganism] = []
        self._alive_worms_by_id: dict[int, Worm] = {}

        self._max_vision = max_vision

    def reset(self) -> dict:
        """Reset the environment for a new episode using procedural generation."""
        Worm._next_id = 0
        FloodOrganism._next_id = 0

        # Generate world from seed
        world_seed = int(self.rng.integers(0, 2**31))
        self.world = generate_world(seed=world_seed,
                                    width=self.width, height=self.height)
        self.terrain = self.world.terrain

        # --- Spawn Lekgolo from formation data ---
        self.worms = []
        self.worms_by_id = {}
        spawns = self.world.spawn_formations

        for (wx, wy) in spawns['lekgolo_worker_positions']:
            worm = Worm(float(wx), float(wy), WormType.WORKER, rng=self.rng)
            self.worms.append(worm)
            self.worms_by_id[worm.id] = worm

        for (tx, ty) in spawns['lekgolo_thinker_positions']:
            worm = Worm(float(tx), float(ty), WormType.THINKER, rng=self.rng)
            self.worms.append(worm)
            self.worms_by_id[worm.id] = worm

        # --- Spawn Flood from formation data ---
        self.flood_list = []
        self.flood_by_id = {}

        for cluster_pos in spawns['flood_clusters']:
            cx, cy = cluster_pos
            for _ in range(FLOOD_SPAWN_CLUSTER_SIZE):
                fx = cx + self.rng.normal(0, 2)
                fy = cy + self.rng.normal(0, 2)
                fx = np.clip(fx, 0, self.width - 1)
                fy = np.clip(fy, 0, self.height - 1)
                flood = FloodOrganism(fx, fy, rng=self.rng)
                self.flood_list.append(flood)
                self.flood_by_id[flood.id] = flood

        for seed_pos in spawns['flood_isolated_seeds']:
            sx, sy = seed_pos
            flood = FloodOrganism(float(sx), float(sy), rng=self.rng)
            self.flood_list.append(flood)
            self.flood_by_id[flood.id] = flood

        # Reset systems
        self.attachment_system = AttachmentSystem()
        self.biomass = INITIAL_BIOMASS
        self.timestep = 0
        self.episode_reward = 0.0
        self.flood_episode_reward = 0.0
        self.thinker_disrupt_timer = 0
        self.active_events = []

        # Reset reward trackers
        self.colony_reward_tracker.reset()
        self.colony_reward_tracker.prev_biomass = self.biomass
        self.flood_reward_tracker.reset()

        # Clear caches
        self._cached_lekgolo_obs = None
        self._cached_flood_obs = None
        self._cached_boosts = None

        # Build initial observations
        observations = self._build_all_observations()
        flood_observations = self._build_all_flood_observations()
        return {
            'lekgolo': observations,
            'flood': flood_observations,
            'world_info': {
                'map_type': self.world.map_type,
                'seed': self.world.seed,
                'modifiers': [m['type'] for m in self.world.modifiers],
                'biomass_fields': len(self.world.biomass_fields),
            }
        }

    def _spawn_flood_at_edge(self):
        """Spawn a Flood organism at a random map edge."""
        if len(self.flood_list) >= MAX_FLOOD_COUNT:
            return
        side = self.rng.integers(0, 4)
        if side == 0:
            x, y = self.rng.uniform(0, self.width), self.rng.uniform(0, 3)
        elif side == 1:
            x, y = self.rng.uniform(0, self.width), self.rng.uniform(self.height - 3, self.height)
        elif side == 2:
            x, y = self.rng.uniform(0, 3), self.rng.uniform(0, self.height)
        else:
            x, y = self.rng.uniform(self.width - 3, self.width), self.rng.uniform(0, self.height)
        flood = FloodOrganism(x, y, rng=self.rng)
        self.flood_list.append(flood)
        self.flood_by_id[flood.id] = flood

    def _rebuild_alive_caches(self):
        """Compute alive_worms and alive_flood lists once per step."""
        self._alive_worms = [w for w in self.worms if w.alive]
        self._alive_flood = [f for f in self.flood_list if f.alive]
        self._alive_worms_by_id = {w.id: w for w in self._alive_worms}

    def _rebuild_spatial_grids(self):
        """Rebuild spatial grids from current entity positions."""
        self.worm_grid.build(self._alive_worms, lambda w: w.x, lambda w: w.y)
        self.flood_grid.build(self._alive_flood, lambda f: f.x, lambda f: f.y)

    def _build_all_observations(self) -> dict:
        """
        Build observation vectors for all alive Lekgolo worms.
        Uses spatial grid for O(k) neighbor lookups instead of O(n²).
        Uses cached thinker boosts computed once per step.
        """
        observations = {}
        alive_worms = self._alive_worms
        alive_flood = self._alive_flood

        # Compute all thinker boosts in one pass
        boosts = self._cached_boosts
        if boosts is None:
            boosts = compute_all_thinker_boosts(alive_worms)
            self._cached_boosts = boosts

        # Apply thinker disrupt event
        disrupt_active = self.thinker_disrupt_timer > 0

        for worm in alive_worms:
            # Use spatial grid to find nearby worms
            worm.nearby_worms = self.worm_grid.query_radius(
                worm.x, worm.y, worm.vision_radius)
            # Filter out self and dead, and verify distance (grid may return extras at cell boundary)
            vision_sq = worm.vision_radius_sq
            worm.nearby_worms = [w for w in worm.nearby_worms
                                 if w.id != worm.id and w.alive and
                                 worm.distance_to_sq(w.x, w.y) <= vision_sq]

            # Use spatial grid for nearby flood
            worm.nearby_enemies = self.flood_grid.query_radius(
                worm.x, worm.y, worm.vision_radius)
            worm.nearby_enemies = [f for f in worm.nearby_enemies
                                   if f.alive and
                                   worm.distance_to_sq(f.x, f.y) <= vision_sq]

            # Terrain lookup
            gx = int(min(max(round(worm.x), 0), self.width - 1))
            gy = int(min(max(round(worm.y), 0), self.height - 1))
            worm.terrain_type = int(self.terrain[gy, gx])

            # Get cached thinker boost
            boost = boosts.get(worm.id, {'attack_accuracy': 0.0,
                                          'move_efficiency': 0.0,
                                          'comm_range': 0.0})
            if disrupt_active and worm.worm_type == 1:
                boost = {k: v * EVENT_THINKER_DISRUPT_BOOST_PENALTY
                         for k, v in boost.items()}

            # Get or allocate observation buffer
            if worm.id not in self._worm_obs_buffer:
                self._worm_obs_buffer[worm.id] = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            obs = build_observation(worm, boost, obs_out=self._worm_obs_buffer[worm.id])

            # Structural strength
            structural = self.attachment_system.compute_structural_strength(worm.id)
            obs[7] = min(structural / 6.0, 1.0)

            # Communication signals (using pre-computed nearby list + adjacency index)
            comm_signals = get_communication_signals(
                worm, self._alive_worms_by_id, self.attachment_system
            )

            # Comm jam modifier
            if self.world:
                wx, wy = worm.x, worm.y
                for mod in self.world.modifiers:
                    if mod['type'] == MODIFIER_COMM_JAM and mod['active']:
                        mx, my = mod['center']
                        if worm.distance_to_sq(mx, my) <= mod['radius'] * mod['radius']:
                            comm_signals *= MODIFIER_COMM_JAM_RANGE_PENALTY

            if np.any(comm_signals != 0):
                for j in range(SIGNAL_DIM):
                    obs[9 + j] = 0.5 * worm.signal[j] + 0.5 * comm_signals[j]

            observations[worm.id] = obs

        self._cached_lekgolo_obs = observations
        return observations

    def _build_all_flood_observations(self) -> dict:
        """
        Build observation vectors for all alive Flood agents.
        Uses spatial grid for O(k) neighbor lookups.
        """
        observations = {}
        alive_worms = self._alive_worms
        alive_flood = self._alive_flood

        for flood in alive_flood:
            # Use spatial grid for nearby worms
            flood.nearby_worms = self.worm_grid.query_radius(
                flood.x, flood.y, FLOOD_VISION)
            flood.nearby_worms = [w for w in flood.nearby_worms
                                  if w.alive and
                                  flood.distance_to_sq(w.x, w.y) <= FLOOD_VISION * FLOOD_VISION]

            # Use spatial grid for nearby flood
            flood.nearby_flood = self.flood_grid.query_radius(
                flood.x, flood.y, FLOOD_VISION)
            flood.nearby_flood = [f for f in flood.nearby_flood
                                  if f.id != flood.id and f.alive and
                                  flood.distance_to_sq(f.x, f.y) <= FLOOD_VISION * FLOOD_VISION]

            # Get or allocate observation buffer
            if flood.id not in self._flood_obs_buffer:
                self._flood_obs_buffer[flood.id] = np.zeros(FLOOD_OBSERVATION_DIM, dtype=np.float32)
            obs = build_flood_observation(
                flood, flood.nearby_worms, flood.nearby_flood,
                self.terrain, self.world.biomass_fields if self.world else [],
                self.worms_by_id, self.attachment_system,
                obs_out=self._flood_obs_buffer[flood.id]
            )
            observations[flood.id] = obs

        self._cached_flood_obs = observations
        return observations

    def _process_events(self):
        """Process scheduled procedural events."""
        if not self.world:
            return

        for event in self.world.event_schedule:
            if event['processed']:
                continue
            if event['trigger_step'] != self.timestep:
                continue

            event['processed'] = True

            if event['type'] == EVENT_FLOOD_SURGE:
                count = event.get('count', 20)
                for _ in range(count):
                    self._spawn_flood_at_edge()

            elif event['type'] == EVENT_RESOURCE_BLOOM:
                self.biomass += event.get('amount', 100)

            elif event['type'] == EVENT_TERRAIN_COLLAPSE:
                radius = event.get('radius', 8)
                ex, ey = event['position']
                for ox in range(-radius, radius + 1):
                    for oy in range(-radius, radius + 1):
                        if ox * ox + oy * oy <= radius * radius:
                            nx, ny = ex + ox, ey + oy
                            if (0 <= nx < self.width and 0 <= ny < self.height and
                                    self.terrain[ny, nx] != TERRAIN_WALL):
                                self.terrain[ny, nx] = TERRAIN_WALL

            elif event['type'] == EVENT_THINKER_DISRUPT:
                self.thinker_disrupt_timer = event.get('duration', 30)

    def step(self) -> dict:
        """Advance the simulation by one timestep."""
        self.timestep += 1

        # Process events
        self._process_events()
        if self.thinker_disrupt_timer > 0:
            self.thinker_disrupt_timer -= 1

        # --- Build alive caches and spatial grids ONCE ---
        self._rebuild_alive_caches()
        self._rebuild_spatial_grids()

        # Compute thinker boosts ONCE per step
        self._cached_boosts = compute_all_thinker_boosts(self._alive_worms)
        disrupt_active = self.thinker_disrupt_timer > 0

        # --- Phase 1: Build observations (once!) ---
        lekgolo_obs = self._build_all_observations()
        flood_obs = self._build_all_flood_observations()

        alive_worms = self._alive_worms
        alive_flood = self._alive_flood

        # --- Phase 2: Lekgolo actions from policy ---
        worm_actions_dict = {}
        worm_log_probs_dict = {}
        worm_values_dict = {}
        worm_type_dict = {}
        worm_dones_dict = {}

        if alive_worms and lekgolo_obs:
            obs_list = []
            worm_ids = []
            worm_types = []
            for worm in alive_worms:
                if worm.id in lekgolo_obs:
                    obs_list.append(lekgolo_obs[worm.id])
                    worm_ids.append(worm.id)
                    worm_types.append(int(worm.worm_type))

            if obs_list:
                obs_batch = torch.FloatTensor(np.array(obs_list))
                type_batch = torch.LongTensor(worm_types)

                with torch.no_grad():
                    actions, log_probs, values, _ = self.lekgolo_policy.get_action(
                        obs_batch, type_batch
                    )

                actions_np = actions.numpy()  # no .cpu() needed on CPU
                log_probs_np = log_probs.numpy()
                values_np = values.numpy().flatten()

                for i, worm_id in enumerate(worm_ids):
                    worm = self.worms_by_id[worm_id]
                    action_dict = decode_action(actions_np[i])
                    boost = self._cached_boosts.get(worm_id,
                                                     {'attack_accuracy': 0.0,
                                                      'move_efficiency': 0.0,
                                                      'comm_range': 0.0})
                    if disrupt_active and worm.worm_type == 1:
                        boost = {k: v * EVENT_THINKER_DISRUPT_BOOST_PENALTY
                                 for k, v in boost.items()}
                    process_action(worm, action_dict, alive_worms, alive_flood,
                                   self.terrain, self.attachment_system,
                                   self.worms_by_id, boost, rng=self.rng)
                    worm_actions_dict[worm_id] = actions_np[i]
                    worm_log_probs_dict[worm_id] = float(log_probs_np[i])
                    worm_values_dict[worm_id] = float(values_np[i])
                    worm_type_dict[worm_id] = int(worm.worm_type)
                    worm_dones_dict[worm_id] = not worm.alive

        # --- Phase 3: Flood actions from policy ---
        flood_actions_dict = {}
        flood_log_probs_dict = {}
        flood_values_dict = {}
        flood_dones_dict = {}

        prev_flood_count = len(alive_flood)
        new_flood = []
        flood_biomass_gained = 0.0

        if alive_flood and flood_obs:
            f_obs_list = []
            f_ids = []
            for f in alive_flood:
                if f.id in flood_obs:
                    f_obs_list.append(flood_obs[f.id])
                    f_ids.append(f.id)

            if f_obs_list:
                f_obs_batch = torch.FloatTensor(np.array(f_obs_list))

                with torch.no_grad():
                    f_actions, f_log_probs, f_values, _ = self.flood_policy.get_action(
                        f_obs_batch
                    )

                f_actions_np = f_actions.numpy()
                f_log_probs_np = f_log_probs.numpy()
                f_values_np = f_values.numpy().flatten()

                for i, f_id in enumerate(f_ids):
                    flood = self.flood_by_id[f_id]
                    action = decode_flood_action(f_actions_np[i])
                    self._execute_flood_action(flood, action, new_flood)

                    flood_actions_dict[f_id] = f_actions_np[i]
                    flood_log_probs_dict[f_id] = float(f_log_probs_np[i])
                    flood_values_dict[f_id] = float(f_values_np[i])
                    flood_dones_dict[f_id] = not flood.alive

        # --- Phase 4: Energy regeneration ---
        for worm in alive_worms:
            worm.regenerate_energy(WORM_ENERGY_REGEN)

        # --- Phase 5: Flood biomass accumulation ---
        for flood in alive_flood:
            flood.biomass += 0.5
            flood.tick_timers()

        # Add new flood
        self.flood_list.extend(new_flood)
        for f in new_flood:
            self.flood_by_id[f.id] = f

        # --- Phase 6: Infection damage over time ---
        for worm in alive_worms:
            if worm.infected and worm.alive:
                worm.infection_timer += 1
                worm.take_damage(FLOOD_INFECTION_DAMAGE * 0.05)
                if self.rng.random() < 0.01:
                    worm.infected = False
                    worm.infection_timer = 0

        # --- Phase 7: Toxic terrain damage ---
        for worm in alive_worms:
            if not worm.alive:
                continue
            gx = int(min(max(round(worm.x), 0), self.width - 1))
            gy = int(min(max(round(worm.y), 0), self.height - 1))
            if self.terrain[gy, gx] == TERRAIN_TOXIC:
                worm.take_damage(TERRAIN_TOXIC_DAMAGE)
        for flood in alive_flood:
            gx = int(min(max(round(flood.x), 0), self.width - 1))
            gy = int(min(max(round(flood.y), 0), self.height - 1))
            if 0 <= gx < self.width and 0 <= gy < self.height:
                if self.terrain[gy, gx] == TERRAIN_TOXIC:
                    flood.take_damage(TERRAIN_TOXIC_DAMAGE)

        # --- Phase 8: Biomass field collection ---
        # BUG FIX: biomass decay modifier was nested inside field loop
        # and used stale loop variable. Now correctly separated.
        if self.world:
            for field in self.world.biomass_fields:
                fx, fy = field['center']
                radius = field['radius']
                radius_sq = radius * radius
                for worm in alive_worms:
                    if worm.alive and worm.distance_to_sq(fx, fy) <= radius_sq:
                        self.biomass += field['rate'] * 0.1

            # Biomass decay modifier — outside field loop, iterates all alive worms
            for mod in self.world.modifiers:
                if mod['type'] == MODIFIER_BIOMASS_DECAY and mod['active']:
                    mx, my = mod['center']
                    mod_r_sq = mod['radius'] * mod['radius']
                    for worm in alive_worms:
                        if worm.alive and worm.distance_to_sq(mx, my) <= mod_r_sq:
                            self.biomass = max(0, self.biomass - MODIFIER_BIOMASS_DECAY_RATE)

        # --- Phase 9: Clean up dead ---
        dead_worms = [w for w in self.worms if not w.alive]
        for worm in dead_worms:
            for nid in list(worm.attachments):
                self.attachment_system.remove_edge(worm.id, nid)
                if nid in self.worms_by_id:
                    self.worms_by_id[nid].attachments.discard(worm.id)
            worm.attachments.clear()

        self.flood_list = [f for f in self.flood_list if f.alive]

        # Spawn new Flood at edges
        for _ in range(FLOOD_SPAWN_RATE):
            self._spawn_flood_at_edge()

        # --- Phase 10: Biomass accumulation ---
        flood_kills = sum(w.flood_kills_this_step for w in alive_worms)
        self.biomass += flood_kills * BIOMASS_PER_FLOOD_KILL
        alive_count = sum(1 for w in self.worms if w.alive)
        self.biomass += alive_count * BIOMASS_PER_TIMESTEP

        # --- Phase 11: Compute rewards ---
        alive_worm_ids = {w.id for w in self.worms if w.alive}
        colony_reward = self.colony_reward_tracker.compute_step_reward(
            self.worms, self.flood_list, self.attachment_system,
            self.terrain, self.biomass
        )

        prev_frags = self.colony_reward_tracker.prev_fragment_count
        current_frags = self.attachment_system.count_fragments(alive_worm_ids)
        colony_fragmented = current_frags > prev_frags

        flood_reward = self.flood_reward_tracker.compute_step_reward(
            self.flood_list, prev_flood_count, colony_fragmented,
            flood_biomass_gained
        )

        self.episode_reward += colony_reward
        self.flood_episode_reward += flood_reward

        # --- Phase 12: Collect rollout data ---
        if lekgolo_obs and worm_actions_dict:
            common_worm_ids = set(lekgolo_obs.keys()) & set(worm_actions_dict.keys())
            if common_worm_ids:
                filtered_obs = {k: lekgolo_obs[k] for k in common_worm_ids}
                filtered_actions = {k: worm_actions_dict[k] for k in common_worm_ids}
                filtered_lp = {k: worm_log_probs_dict.get(k, 0.0) for k in common_worm_ids}
                filtered_vals = {k: worm_values_dict.get(k, 0.0) for k in common_worm_ids}
                filtered_dones = {k: worm_dones_dict.get(k, True) for k in common_worm_ids}
                filtered_types = {k: worm_type_dict.get(k, 0) for k in common_worm_ids}
                self.lekgolo_trainer.collect_step(
                    filtered_obs, filtered_actions, filtered_lp,
                    filtered_vals, colony_reward, filtered_dones, filtered_types
                )
        if flood_obs and flood_actions_dict:
            common_flood_ids = set(flood_obs.keys()) & set(flood_actions_dict.keys())
            if common_flood_ids:
                filtered_fobs = {k: flood_obs[k] for k in common_flood_ids}
                filtered_factions = {k: flood_actions_dict[k] for k in common_flood_ids}
                filtered_flp = {k: flood_log_probs_dict.get(k, 0.0) for k in common_flood_ids}
                filtered_fvals = {k: flood_values_dict.get(k, 0.0) for k in common_flood_ids}
                filtered_fdones = {k: flood_dones_dict.get(k, True) for k in common_flood_ids}
                self.flood_trainer.collect_step(
                    filtered_fobs, filtered_factions, filtered_flp,
                    filtered_fvals, flood_reward, filtered_fdones,
                    {f_id: 0 for f_id in common_flood_ids}
                )

        # --- Phase 13: Reset per-step trackers ---
        for worm in alive_worms:
            worm.damage_blocked_this_step = 0.0
            worm.flood_kills_this_step = 0
            worm.local_damage_taken = 0.0
        for flood in self.flood_list:
            flood.reset_step_trackers()

        self.attachment_system.tick_edges()

        # --- Check done ---
        done = (
            self.timestep >= MAX_STEPS_PER_EPISODE or
            not any(w.alive for w in self.worms)
        )

        # Reuse cached observations for return — NO second observation build!
        # The observations were already built at Phase 1 and are still valid
        # for the next step's input (they represent current state).
        new_lekgolo_obs = self._cached_lekgolo_obs or lekgolo_obs
        new_flood_obs = self._cached_flood_obs or flood_obs

        info = {
            'timestep': self.timestep,
            'alive_worms': alive_count,
            'alive_thinkers': sum(1 for w in self.worms if w.alive and w.worm_type == WormType.THINKER),
            'alive_workers': sum(1 for w in self.worms if w.alive and w.worm_type == WormType.WORKER),
            'alive_flood': len([f for f in self.flood_list if f.alive]),
            'biomass': self.biomass,
            'colony_fragments': current_frags,
            'attachments': len(self.attachment_system.edges),
            'infected_worms': sum(1 for w in self.worms if w.alive and w.infected),
            'episode_reward': self.episode_reward,
            'flood_episode_reward': self.flood_episode_reward,
            'colony_reward_breakdown': self.colony_reward_tracker.reward_breakdown,
            'flood_reward_breakdown': self.flood_reward_tracker.reward_breakdown,
            'map_type': self.world.map_type if self.world else 'unknown',
            'thinker_disrupt_active': self.thinker_disrupt_timer > 0,
        }

        # Clear caches for next step
        self._cached_lekgolo_obs = None
        self._cached_flood_obs = None
        self._cached_boosts = None

        return {
            'lekgolo_observations': new_lekgolo_obs,
            'flood_observations': new_flood_obs,
            'colony_reward': colony_reward,
            'flood_reward': flood_reward,
            'done': done,
            'info': info,
        }

    def _execute_flood_action(self, flood: FloodOrganism, action: dict,
                              new_flood: list):
        """Execute a single Flood agent action."""
        if not flood.alive:
            return

        action_type = action['type']
        params = action['params']

        if action_type == 0:  # Move
            dx, dy = params[0] * FLOOD_SPEED, params[1] * FLOOD_SPEED
            new_x = flood.x + dx
            new_y = flood.y + dy
            gx = int(min(max(round(new_x), 0), self.width - 1))
            gy = int(min(max(round(new_y), 0), self.height - 1))
            if is_passable(self.terrain, gx, gy):
                flood.x = min(max(new_x, 0), self.width - 1)
                flood.y = min(max(new_y, 0), self.height - 1)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    flood.orientation = math.atan2(dy, dx)

        elif action_type == 1:  # Attack
            if flood.nearby_worms:
                target_idx = int(min(max(
                    int(round(params[0] * len(flood.nearby_worms))),
                    0), len(flood.nearby_worms) - 1))
                target = flood.nearby_worms[target_idx]
                if target.alive:
                    d_sq = flood.distance_to_sq(target.x, target.y)
                    if d_sq <= _FLOOD_ATTACK_RANGE_SQ:
                        damage = FLOOD_ATTACK_DAMAGE
                        if self.world:
                            for mod in self.world.modifiers:
                                if (mod['type'] == MODIFIER_INFECTION_FOG and
                                        mod['active']):
                                    mx, my = mod['center']
                                    if flood.distance_to_sq(mx, my) <= mod['radius'] * mod['radius']:
                                        damage *= MODIFIER_INFECTION_FOG_FLOOD_BOOST
                        reduction = self.attachment_system.compute_damage_reduction(target.id)
                        actual_damage = damage * (1.0 - reduction)
                        blocked = damage * reduction
                        target.take_damage(actual_damage)
                        target.damage_blocked_this_step += blocked
                        flood.damage_dealt_this_step += actual_damage
                        if not target.alive:
                            flood.kills_this_step += 1

        elif action_type == 2:  # Infect
            if flood.nearby_worms:
                target_idx = int(min(max(
                    int(round(params[0] * len(flood.nearby_worms))),
                    0), len(flood.nearby_worms) - 1))
                target = flood.nearby_worms[target_idx]
                if target.alive and not target.infected:
                    d_sq = flood.distance_to_sq(target.x, target.y)
                    if d_sq <= _FLOOD_ATTACK_RANGE_SQ:
                        if self.rng.random() < 0.3:
                            target.infected = True
                            target.infection_timer += 1
                            target.take_damage(FLOOD_INFECTION_DAMAGE * 0.1)
                            flood.infections_this_step += 1
                            self.flood_reward_tracker.compute_infection_reward(
                                int(target.worm_type))
                            flood.biomass += BIOMASS_COST_WORKER * 0.5

        elif action_type == 3:  # Split/Spawn
            if flood.can_reproduce() and len(self.flood_list) + len(new_flood) < MAX_FLOOD_COUNT:
                offset_x = self.rng.normal(0, 2)
                offset_y = self.rng.normal(0, 2)
                child = FloodOrganism(
                    min(max(flood.x + offset_x, 0), self.width - 1),
                    min(max(flood.y + offset_y, 0), self.height - 1),
                    rng=self.rng
                )
                new_flood.append(child)
                flood.biomass -= FLOOD_REPRODUCE_BIOMASS_COST
                flood.reproduce_timer = FLOOD_REPRODUCE_INTERVAL

        elif action_type == 4:  # Signal
            for i in range(min(len(params), FLOOD_SIGNAL_DIM)):
                flood.signal[i] = max(-1.0, min(1.0, float(params[i])))

    def run_episode(self, max_steps: int | None = None,
                    render: bool = False,
                    train: bool = True) -> dict:
        """Run a full episode with training for both sides."""
        max_steps = max_steps or MAX_STEPS_PER_EPISODE
        result = self.reset()
        observations = result.get('lekgolo', {})
        flood_observations = result.get('flood', {})

        episode_stats = []
        update_interval = PPO_ROLLOUT_LENGTH

        for step in range(max_steps):
            step_result = self.step()

            if render and step % 10 == 0:
                os.makedirs(DEFAULT_FRAME_DIR, exist_ok=True)
                self.render(save_path=os.path.join(DEFAULT_FRAME_DIR, f'frame_{step:06d}.png'))

            # Periodic PPO updates
            if train:
                if len(self.lekgolo_trainer.buffer) >= update_interval:
                    stats = self.lekgolo_trainer.update(0.0)
                    episode_stats.append({'lekgolo': stats})
                if len(self.flood_trainer.buffer) >= update_interval:
                    stats = self.flood_trainer.update(0.0)
                    episode_stats.append({'flood': stats})

            if step_result['done']:
                break

        # Final updates
        if train:
            if len(self.lekgolo_trainer.buffer) > 0:
                self.lekgolo_trainer.update(0.0)
            if len(self.flood_trainer.buffer) > 0:
                self.flood_trainer.update(0.0)

        return {
            'episode_reward': self.episode_reward,
            'flood_episode_reward': self.flood_episode_reward,
            'timesteps': self.timestep,
            'training_stats': episode_stats,
            'final_info': step_result.get('info', {}),
            'colony_reward_breakdown': self.colony_reward_tracker.reward_breakdown,
            'flood_reward_breakdown': self.flood_reward_tracker.reward_breakdown,
            'map_type': self.world.map_type if self.world else 'unknown',
        }

    def render(self, save_path: str | None = None, show: bool = False):
        """Render the current state of the simulation."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        setup_matplotlib_fonts()

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        terrain_colors = {
            TERRAIN_FLAT: [0.2, 0.5, 0.2],
            TERRAIN_ROUGH: [0.5, 0.4, 0.2],
            TERRAIN_WALL: [0.3, 0.3, 0.3],
            TERRAIN_TOXIC: [0.6, 0.2, 0.6],
            TERRAIN_HIGHGROUND: [0.4, 0.6, 0.4],
        }
        terrain_img = np.zeros((self.height, self.width, 3))
        for t_val, color in terrain_colors.items():
            terrain_img[self.terrain == t_val] = color
        ax.imshow(terrain_img, extent=[0, self.width, 0, self.height], alpha=0.6)

        if self.world:
            for field in self.world.biomass_fields:
                fx, fy = field['center']
                r = field['radius']
                color = 'gold' if field['is_contested'] else 'lightgreen'
                circle = plt.Circle((fx, fy), r, color=color, alpha=0.15)
                ax.add_patch(circle)

        if self.world:
            mod_colors = {
                MODIFIER_INFECTION_FOG: 'purple',
                MODIFIER_COMM_JAM: 'orange',
                MODIFIER_BIOMASS_DECAY: 'brown',
                MODIFIER_COLLAPSING: 'red',
            }
            for mod in self.world.modifiers:
                if mod['active']:
                    mx, my = mod['center']
                    r = mod['radius']
                    color = mod_colors.get(mod['type'], 'gray')
                    circle = plt.Circle((mx, my), r, color=color, alpha=0.1)
                    ax.add_patch(circle)

        alive_worms_map = {w.id: w for w in self.worms if w.alive}
        for (a, b) in self.attachment_system.edges:
            if a in alive_worms_map and b in alive_worms_map:
                wa, wb = alive_worms_map[a], alive_worms_map[b]
                ax.plot([wa.x, wb.x], [wa.y, wb.y], 'y-', alpha=0.3, linewidth=0.5)

        workers = [w for w in self.worms if w.alive and w.worm_type == WormType.WORKER]
        if workers:
            wx = [w.x for w in workers]
            wy = [w.y for w in workers]
            colors = ['red' if w.infected else 'lime' for w in workers]
            ax.scatter(wx, wy, c=colors, s=8, alpha=0.7, zorder=3)

        thinkers = [w for w in self.worms if w.alive and w.worm_type == WormType.THINKER]
        if thinkers:
            tx = [w.x for w in thinkers]
            ty = [w.y for w in thinkers]
            colors_t = ['red' if w.infected else 'cyan' for w in thinkers]
            ax.scatter(tx, ty, c=colors_t, s=30, marker='*', alpha=0.9, zorder=4)

        alive_flood = [f for f in self.flood_list if f.alive]
        if alive_flood:
            fx = [f.x for f in alive_flood]
            fy = [f.y for f in alive_flood]
            ax.scatter(fx, fy, c='darkred', s=6, alpha=0.6, marker='x', zorder=2)

        alive_w = len(alive_worms_map)
        alive_t = sum(1 for w in thinkers if w.alive)
        infected = sum(1 for w in self.worms if w.alive and w.infected)
        attachments = len(self.attachment_system.edges)
        map_type = self.world.map_type if self.world else '?'

        info_text = (
            f"Step: {self.timestep}  |  "
            f"Map: {map_type}  |  "
            f"Worms: {alive_w} (T:{alive_t})  |  "
            f"Flood: {len(alive_flood)}  |  "
            f"Infected: {infected}  |  "
            f"Links: {attachments}  |  "
            f"Biomass: {self.biomass:.0f}"
        )
        ax.set_title(info_text, fontsize=10)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def save_checkpoint(self, path: str):
        """Save both policies and optimizers."""
        torch.save({
            'lekgolo_worker': self.lekgolo_policy.worker_policy.state_dict(),
            'lekgolo_thinker': self.lekgolo_policy.thinker_policy.state_dict(),
            'flood_policy': self.flood_policy.policy_net.state_dict(),
            'lekgolo_optimizer': self.lekgolo_trainer.optimizer.state_dict(),
            'flood_optimizer': self.flood_trainer.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load policies from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.lekgolo_policy.worker_policy.load_state_dict(checkpoint['lekgolo_worker'])
        self.lekgolo_policy.thinker_policy.load_state_dict(checkpoint['lekgolo_thinker'])
        self.flood_policy.policy_net.load_state_dict(checkpoint['flood_policy'])
        self.lekgolo_trainer.optimizer.load_state_dict(checkpoint['lekgolo_optimizer'])
        self.flood_trainer.optimizer.load_state_dict(checkpoint['flood_optimizer'])
