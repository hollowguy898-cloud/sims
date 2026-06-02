"""
Main Lekgolo Colony simulation environment.

Uses the full procedural generation pipeline and treats both sides
as RL agents:
  Lekgolo colony: distributed organism (structure + coordination + thinker protection)
  Flood: self-replicating parasite colony (replication + disruption + spread)

Both sides learn. Both sides adapt. Neither is scripted.
"""
import numpy as np
import torch
from collections import defaultdict

from config import (
    MAP_WIDTH, MAP_HEIGHT, MAX_STEPS_PER_EPISODE,
    NUM_WORMS_INITIAL, NUM_THINKERS_INITIAL, NUM_FLOOD_INITIAL,
    FLOOD_SPAWN_RATE, MAX_FLOOD_COUNT,
    WORM_MAX_ENERGY, WORM_ENERGY_REGEN, WORM_ATTACK_RANGE,
    WORM_ATTACK_DAMAGE_WORKER, WORM_ATTACK_DAMAGE_THINKER,
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
)
from world_gen import generate_world, is_passable, movement_cost, WorldData
from worm import Worm, WormType
from flood import (FloodOrganism, build_flood_observation,
                   decode_flood_action, FLOOD_OBSERVATION_DIM)
from attachment_system import AttachmentSystem, compute_thinker_boost
from sensors import build_observation, get_communication_signals, OBSERVATION_DIM
from actions import decode_action, process_action, ACTION_VECTOR_DIM
from rewards import ColonyRewardTracker, FloodRewardTracker
from network import ColonySharedPolicy, FloodPolicy
from ppo_trainer import PPOTrainer


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

        # Workers
        for (wx, wy) in spawns['lekgolo_worker_positions']:
            worm = Worm(float(wx), float(wy), WormType.WORKER)
            self.worms.append(worm)
            self.worms_by_id[worm.id] = worm

        # Thinkers
        for (tx, ty) in spawns['lekgolo_thinker_positions']:
            worm = Worm(float(tx), float(ty), WormType.THINKER)
            self.worms.append(worm)
            self.worms_by_id[worm.id] = worm

        # --- Spawn Flood from formation data ---
        self.flood_list = []
        self.flood_by_id = {}

        # Cluster spawns
        total_cluster_flood = 0
        for cluster_pos in spawns['flood_clusters']:
            cx, cy = cluster_pos
            for _ in range(FLOOD_SPAWN_CLUSTER_SIZE):
                fx = cx + self.rng.normal(0, 2)
                fy = cy + self.rng.normal(0, 2)
                fx = np.clip(fx, 0, self.width - 1)
                fy = np.clip(fy, 0, self.height - 1)
                flood = FloodOrganism(fx, fy)
                self.flood_list.append(flood)
                self.flood_by_id[flood.id] = flood
                total_cluster_flood += 1

        # Isolated seeds
        for seed_pos in spawns['flood_isolated_seeds']:
            sx, sy = seed_pos
            flood = FloodOrganism(float(sx), float(sy))
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
        flood = FloodOrganism(x, y)
        self.flood_list.append(flood)
        self.flood_by_id[flood.id] = flood

    def _build_all_observations(self) -> dict:
        """Build observation vectors for all alive Lekgolo worms."""
        observations = {}
        alive_worms = [w for w in self.worms if w.alive]
        alive_flood = [f for f in self.flood_list if f.alive]

        for worm in alive_worms:
            worm.nearby_worms = []
            worm.nearby_enemies = []

            for other in alive_worms:
                if other.id == worm.id:
                    continue
                d = worm.distance_to(other.x, other.y)
                if d <= worm.vision_radius:
                    worm.nearby_worms.append(other)

            for enemy in alive_flood:
                d = worm.distance_to(enemy.x, enemy.y)
                if d <= worm.vision_radius:
                    worm.nearby_enemies.append(enemy)

            gx = int(np.clip(np.round(worm.x), 0, self.width - 1))
            gy = int(np.clip(np.round(worm.y), 0, self.height - 1))
            worm.terrain_type = int(self.terrain[gy, gx])

            # Highground vision bonus
            effective_vision = worm.vision_radius
            if self.terrain[gy, gx] == TERRAIN_HIGHGROUND:
                effective_vision += TERRAIN_HIGHGROUND_VISION_BONUS

            boost = compute_thinker_boost(worm, alive_worms)

            # Thinker disrupt event penalty
            if self.thinker_disrupt_timer > 0 and worm.worm_type == 1:
                boost = {k: v * EVENT_THINKER_DISRUPT_BOOST_PENALTY
                         for k, v in boost.items()}

            obs = build_observation(worm, alive_worms, alive_flood,
                                   self.terrain, boost)

            structural = self.attachment_system.compute_structural_strength(worm.id)
            obs[7] = min(structural / 6.0, 1.0)

            comm_signals = get_communication_signals(
                worm, alive_worms, self.attachment_system
            )
            # Comm jam modifier reduces range
            if self.world:
                for mod in self.world.modifiers:
                    if mod['type'] == MODIFIER_COMM_JAM and mod['active']:
                        mx, my = mod['center']
                        if worm.distance_to(mx, my) <= mod['radius']:
                            comm_signals *= MODIFIER_COMM_JAM_RANGE_PENALTY

            if np.any(comm_signals != 0):
                obs[9:9 + SIGNAL_DIM] = 0.5 * worm.signal + 0.5 * comm_signals

            observations[worm.id] = obs
        return observations

    def _build_all_flood_observations(self) -> dict:
        """Build observation vectors for all alive Flood agents."""
        observations = {}
        alive_worms = [w for w in self.worms if w.alive]
        alive_flood = [f for f in self.flood_list if f.alive]

        for flood in alive_flood:
            # Find nearby worms
            flood.nearby_worms = []
            for worm in alive_worms:
                d = flood.distance_to(worm.x, worm.y)
                if d <= FLOOD_VISION:
                    flood.nearby_worms.append(worm)

            # Find nearby Flood
            flood.nearby_flood = []
            for other in alive_flood:
                if other.id == flood.id or not other.alive:
                    continue
                d = flood.distance_to(other.x, other.y)
                if d <= FLOOD_VISION:
                    flood.nearby_flood.append(other)

            obs = build_flood_observation(
                flood, flood.nearby_worms, flood.nearby_flood,
                self.terrain, self.world.biomass_fields if self.world else [],
                self.worms_by_id, self.attachment_system
            )
            observations[flood.id] = obs
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

        # --- Phase 1: Build observations ---
        lekgolo_obs = self._build_all_observations()
        flood_obs = self._build_all_flood_observations()

        # --- Phase 2: Lekgolo actions from policy ---
        alive_worms = [w for w in self.worms if w.alive]
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
                obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
                type_batch = torch.LongTensor(worm_types).to(self.device)

                with torch.no_grad():
                    actions, log_probs, values, _ = self.lekgolo_policy.get_action(
                        obs_batch, type_batch
                    )

                actions_np = actions.cpu().numpy()
                log_probs_np = log_probs.cpu().numpy()
                values_np = values.cpu().numpy().flatten()

                for i, worm_id in enumerate(worm_ids):
                    worm = self.worms_by_id[worm_id]
                    action_dict = decode_action(actions_np[i])
                    boost = compute_thinker_boost(worm, alive_worms)
                    if self.thinker_disrupt_timer > 0 and worm.worm_type == 1:
                        boost = {k: v * EVENT_THINKER_DISRUPT_BOOST_PENALTY
                                 for k, v in boost.items()}
                    process_action(worm, action_dict, alive_worms, self.flood_list,
                                   self.terrain, self.attachment_system,
                                   self.worms_by_id, boost)
                    worm_actions_dict[worm_id] = actions_np[i]
                    worm_log_probs_dict[worm_id] = float(log_probs_np[i])
                    worm_values_dict[worm_id] = float(values_np[i])
                    worm_type_dict[worm_id] = int(worm.worm_type)
                    worm_dones_dict[worm_id] = not worm.alive

        # --- Phase 3: Flood actions from policy ---
        alive_flood_agents = [f for f in self.flood_list if f.alive]
        flood_actions_dict = {}
        flood_log_probs_dict = {}
        flood_values_dict = {}
        flood_dones_dict = {}

        prev_flood_count = len(alive_flood_agents)
        new_flood = []
        flood_biomass_gained = 0.0

        if alive_flood_agents and flood_obs:
            f_obs_list = []
            f_ids = []
            for f in alive_flood_agents:
                if f.id in flood_obs:
                    f_obs_list.append(flood_obs[f.id])
                    f_ids.append(f.id)

            if f_obs_list:
                f_obs_batch = torch.FloatTensor(np.array(f_obs_list)).to(self.device)

                with torch.no_grad():
                    f_actions, f_log_probs, f_values, _ = self.flood_policy.get_action(
                        f_obs_batch
                    )

                f_actions_np = f_actions.cpu().numpy()
                f_log_probs_np = f_log_probs.cpu().numpy()
                f_values_np = f_values.cpu().numpy().flatten()

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
        for flood in self.flood_list:
            if flood.alive:
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
                if np.random.random() < 0.01:
                    worm.infected = False
                    worm.infection_timer = 0

        # --- Phase 7: Toxic terrain damage ---
        for worm in alive_worms:
            if not worm.alive:
                continue
            gx = int(np.clip(np.round(worm.x), 0, self.width - 1))
            gy = int(np.clip(np.round(worm.y), 0, self.height - 1))
            if self.terrain[gy, gx] == TERRAIN_TOXIC:
                worm.take_damage(TERRAIN_TOXIC_DAMAGE)
        for flood in self.flood_list:
            if not flood.alive:
                continue
            gx = int(np.clip(np.round(flood.x), 0, self.width - 1))
            gy = int(np.clip(np.round(flood.y), 0, self.height - 1))
            if 0 <= gx < self.width and 0 <= gy < self.height:
                if self.terrain[gy, gx] == TERRAIN_TOXIC:
                    flood.take_damage(TERRAIN_TOXIC_DAMAGE)

        # --- Phase 8: Biomass field collection ---
        if self.world:
            for field in self.world.biomass_fields:
                fx, fy = field['center']
                radius = field['radius']
                for worm in alive_worms:
                    if worm.alive and worm.distance_to(fx, fy) <= radius:
                        self.biomass += field['rate'] * 0.1
                # Biomass decay modifier
                for mod in self.world.modifiers:
                    if mod['type'] == MODIFIER_BIOMASS_DECAY and mod['active']:
                        mx, my = mod['center']
                        if worm.distance_to(mx, my) <= mod['radius']:
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
            # Only collect for IDs present in both obs and actions
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

        new_lekgolo_obs = self._build_all_observations()
        new_flood_obs = self._build_all_flood_observations()

        info = {
            'timestep': self.timestep,
            'alive_worms': sum(1 for w in self.worms if w.alive),
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
            gx = int(np.clip(np.round(new_x), 0, self.width - 1))
            gy = int(np.clip(np.round(new_y), 0, self.height - 1))
            if is_passable(self.terrain, gx, gy):
                flood.x = np.clip(new_x, 0, self.width - 1)
                flood.y = np.clip(new_y, 0, self.height - 1)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    flood.orientation = np.arctan2(dy, dx)

        elif action_type == 1:  # Attack
            if flood.nearby_worms:
                target_idx = int(np.clip(
                    int(np.round(params[0] * len(flood.nearby_worms))),
                    0, len(flood.nearby_worms) - 1))
                target = flood.nearby_worms[target_idx]
                if target.alive:
                    d = flood.distance_to(target.x, target.y)
                    if d <= FLOOD_ATTACK_RANGE:
                        damage = FLOOD_ATTACK_DAMAGE
                        # Infection fog boost
                        if self.world:
                            for mod in self.world.modifiers:
                                if (mod['type'] == MODIFIER_INFECTION_FOG and
                                        mod['active']):
                                    mx, my = mod['center']
                                    if flood.distance_to(mx, my) <= mod['radius']:
                                        damage *= MODIFIER_INFECTION_FOG_FLOOD_BOOST
                        # Structural damage reduction
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
                target_idx = int(np.clip(
                    int(np.round(params[0] * len(flood.nearby_worms))),
                    0, len(flood.nearby_worms) - 1))
                target = flood.nearby_worms[target_idx]
                if target.alive and not target.infected:
                    d = flood.distance_to(target.x, target.y)
                    if d <= FLOOD_ATTACK_RANGE:
                        # 1:1 infection rule: 30% base chance
                        if np.random.random() < 0.3:
                            target.infected = True
                            target.infection_timer += 1
                            target.take_damage(FLOOD_INFECTION_DAMAGE * 0.1)
                            flood.infections_this_step += 1
                            self.flood_reward_tracker.compute_infection_reward(
                                int(target.worm_type))
                            # Flood gains biomass from infection
                            flood.biomass += BIOMASS_COST_WORKER * 0.5

        elif action_type == 3:  # Split/Spawn
            if flood.can_reproduce() and len(self.flood_list) + len(new_flood) < MAX_FLOOD_COUNT:
                offset_x = self.rng.normal(0, 2)
                offset_y = self.rng.normal(0, 2)
                child = FloodOrganism(
                    np.clip(flood.x + offset_x, 0, self.width - 1),
                    np.clip(flood.y + offset_y, 0, self.height - 1)
                )
                new_flood.append(child)
                flood.biomass -= FLOOD_REPRODUCE_BIOMASS_COST
                flood.reproduce_timer = FLOOD_REPRODUCE_INTERVAL

        elif action_type == 4:  # Signal
            flood.signal = np.clip(params[:FLOOD_SIGNAL_DIM], -1.0, 1.0).astype(np.float32)

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
                self.render(save_path=f'/home/z/my-project/download/frame_{step:06d}.png')

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
        import matplotlib.font_manager as fm

        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Draw terrain (5 types now)
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

        # Draw biomass fields
        if self.world:
            for field in self.world.biomass_fields:
                fx, fy = field['center']
                r = field['radius']
                color = 'gold' if field['is_contested'] else 'lightgreen'
                circle = plt.Circle((fx, fy), r, color=color, alpha=0.15)
                ax.add_patch(circle)

        # Draw modifiers
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

        # Draw attachments
        alive_worms_map = {w.id: w for w in self.worms if w.alive}
        for (a, b) in self.attachment_system.edges:
            if a in alive_worms_map and b in alive_worms_map:
                wa, wb = alive_worms_map[a], alive_worms_map[b]
                ax.plot([wa.x, wb.x], [wa.y, wb.y], 'y-', alpha=0.3, linewidth=0.5)

        # Draw workers
        workers = [w for w in self.worms if w.alive and w.worm_type == WormType.WORKER]
        if workers:
            wx = [w.x for w in workers]
            wy = [w.y for w in workers]
            colors = ['red' if w.infected else 'lime' for w in workers]
            ax.scatter(wx, wy, c=colors, s=8, alpha=0.7, zorder=3)

        # Draw thinkers
        thinkers = [w for w in self.worms if w.alive and w.worm_type == WormType.THINKER]
        if thinkers:
            tx = [w.x for w in thinkers]
            ty = [w.y for w in thinkers]
            colors_t = ['red' if w.infected else 'cyan' for w in thinkers]
            ax.scatter(tx, ty, c=colors_t, s=30, marker='*', alpha=0.9, zorder=4)

        # Draw Flood
        alive_flood = [f for f in self.flood_list if f.alive]
        if alive_flood:
            fx = [f.x for f in alive_flood]
            fy = [f.y for f in alive_flood]
            ax.scatter(fx, fy, c='darkred', s=6, alpha=0.6, marker='x', zorder=2)

        # Info text
        alive_w = sum(1 for w in self.worms if w.alive)
        alive_t = sum(1 for w in self.worms if w.alive and w.worm_type == WormType.THINKER)
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
