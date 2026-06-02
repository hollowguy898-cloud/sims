"""
Main Lekgolo Colony simulation environment.

This is the central module that:
- Manages the world (terrain, worms, Flood)
- Steps the simulation forward
- Connects sensors -> policy -> actions -> physics
- Computes rewards
- Provides the training loop
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
    ATTACHMENT_MAX_DISTANCE, THINKER_BOOST_RADIUS,
    PPO_ROLLOUT_LENGTH,
)
from terrain import generate_terrain, is_passable, movement_cost
from worm import Worm, WormType
from flood import FloodOrganism
from attachment_system import AttachmentSystem, compute_thinker_boost
from sensors import build_observation, get_communication_signals, OBSERVATION_DIM
from actions import (
    decode_action, process_action, execute_move, execute_attack,
    execute_signal, ACTION_VECTOR_DIM
)
from rewards import ColonyRewardTracker, FloodRewardTracker
from network import ColonySharedPolicy
from ppo_trainer import PPOTrainer


class LekgoloEnvironment:
    """
    The main simulation environment.

    Manages the world state and provides step/reset interface
    compatible with RL training.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT

        # World state
        self.terrain = generate_terrain(self.width, self.height, seed)
        self.worms: list[Worm] = []
        self.flood_list: list[FloodOrganism] = []
        self.worms_by_id: dict[int, Worm] = {}
        self.flood_by_id: dict[int, FloodOrganism] = {}
        self.attachment_system = AttachmentSystem()

        # Colony state
        self.biomass = INITIAL_BIOMASS
        self.timestep = 0
        self.episode_reward = 0.0

        # Reward trackers
        self.colony_reward_tracker = ColonyRewardTracker()
        self.flood_reward_tracker = FloodRewardTracker()

        # RL components
        self.device = 'cpu'
        self.policy = ColonySharedPolicy(OBSERVATION_DIM).to(self.device)
        self.trainer = PPOTrainer(self.policy, self.device)

    def reset(self) -> dict:
        """
        Reset the environment for a new episode.

        Returns:
            Initial observations dict {worm_id: obs_array}
        """
        # Reset IDs
        Worm._next_id = 0
        FloodOrganism._next_id = 0

        # Regenerate terrain
        self.terrain = generate_terrain(self.width, self.height,
                                        self.rng.integers(0, 100000))

        # Spawn worms near center of map
        center_x, center_y = self.width / 2, self.height / 2
        self.worms = []
        self.worms_by_id = {}

        # Workers
        num_workers = NUM_WORMS_INITIAL - NUM_THINKERS_INITIAL
        for i in range(num_workers):
            x = center_x + self.rng.normal(0, 5)
            y = center_y + self.rng.normal(0, 5)
            x = np.clip(x, 2, self.width - 3)
            y = np.clip(y, 2, self.height - 3)
            worm = Worm(x, y, WormType.WORKER)
            self.worms.append(worm)
            self.worms_by_id[worm.id] = worm

        # Thinkers
        for i in range(NUM_THINKERS_INITIAL):
            x = center_x + self.rng.normal(0, 3)
            y = center_y + self.rng.normal(0, 3)
            x = np.clip(x, 2, self.width - 3)
            y = np.clip(y, 2, self.height - 3)
            worm = Worm(x, y, WormType.THINKER)
            self.worms.append(worm)
            self.worms_by_id[worm.id] = worm

        # Spawn Flood at map edges
        self.flood_list = []
        self.flood_by_id = {}
        for i in range(NUM_FLOOD_INITIAL):
            self._spawn_flood_at_edge()

        # Reset attachment system
        self.attachment_system = AttachmentSystem()

        # Reset state
        self.biomass = INITIAL_BIOMASS
        self.timestep = 0
        self.episode_reward = 0.0

        # Reset reward trackers
        self.colony_reward_tracker.reset()
        self.colony_reward_tracker.prev_biomass = self.biomass
        self.flood_reward_tracker.reset()

        # Build initial observations
        observations = self._build_all_observations()
        return observations

    def _spawn_flood_at_edge(self):
        """Spawn a Flood organism at a random map edge."""
        if len(self.flood_list) >= MAX_FLOOD_COUNT:
            return

        side = self.rng.integers(0, 4)
        if side == 0:  # top
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, 3)
        elif side == 1:  # bottom
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(self.height - 3, self.height)
        elif side == 2:  # left
            x = self.rng.uniform(0, 3)
            y = self.rng.uniform(0, self.height)
        else:  # right
            x = self.rng.uniform(self.width - 3, self.width)
            y = self.rng.uniform(0, self.height)

        flood = FloodOrganism(x, y)
        self.flood_list.append(flood)
        self.flood_by_id[flood.id] = flood

    def _build_all_observations(self) -> dict:
        """Build observation vectors for all alive worms."""
        observations = {}
        alive_worms = [w for w in self.worms if w.alive]
        alive_flood = [f for f in self.flood_list if f.alive]

        for worm in alive_worms:
            # Update sensor caches
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

            # Get terrain at worm position
            gx = int(np.clip(np.round(worm.x), 0, self.width - 1))
            gy = int(np.clip(np.round(worm.y), 0, self.height - 1))
            worm.terrain_type = int(self.terrain[gy, gx])

            # Compute thinker boost
            boost = compute_thinker_boost(worm, alive_worms)

            # Build observation
            obs = build_observation(worm, alive_worms, alive_flood,
                                   self.terrain, boost)

            # Fill in structural strength
            structural = self.attachment_system.compute_structural_strength(worm.id)
            obs[7] = min(structural / 6.0, 1.0)  # normalized by max attachments

            # Incorporate communication signals
            comm_signals = get_communication_signals(
                worm, alive_worms, self.attachment_system
            )
            # Blend own signal with received signals
            if np.any(comm_signals != 0):
                obs[9:9 + SIGNAL_DIM] = 0.5 * worm.signal + 0.5 * comm_signals

            observations[worm.id] = obs

        return observations

    def step(self) -> dict:
        """
        Advance the simulation by one timestep.

        Returns:
            dict with:
                'observations': {worm_id: obs_array}
                'colony_reward': float
                'flood_reward': float
                'done': bool
                'info': dict
        """
        self.timestep += 1

        # --- Phase 1: Build observations ---
        observations = self._build_all_observations()

        # --- Phase 2: Get actions from policy ---
        alive_worms = [w for w in self.worms if w.alive]
        if not alive_worms:
            return {
                'observations': {},
                'colony_reward': 0.0,
                'flood_reward': 0.0,
                'done': True,
                'info': {'reason': 'all_worms_dead'},
            }

        # Batch inference
        obs_list = []
        worm_ids = []
        worm_types = []

        for worm in alive_worms:
            if worm.id in observations:
                obs_list.append(observations[worm.id])
                worm_ids.append(worm.id)
                worm_types.append(int(worm.worm_type))

        if not obs_list:
            return {
                'observations': observations,
                'colony_reward': 0.0,
                'flood_reward': 0.0,
                'done': self.timestep >= MAX_STEPS_PER_EPISODE,
                'info': {},
            }

        obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
        worm_type_batch = torch.LongTensor(worm_types).to(self.device)

        with torch.no_grad():
            actions, log_probs, values, entropies = self.policy.get_action(
                obs_batch, worm_type_batch
            )

        # Convert to numpy
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        # --- Phase 3: Execute worm actions ---
        action_results = {}
        actions_dict = {}
        log_probs_dict = {}
        values_dict = {}
        worm_type_dict = {}
        dones_dict = {}

        for i, worm_id in enumerate(worm_ids):
            worm = self.worms_by_id[worm_id]
            action_vector = actions_np[i]
            action_dict = decode_action(action_vector)

            # Get thinker boost for this worm
            boost = compute_thinker_boost(worm, alive_worms)

            # Process the action
            success = process_action(
                worm, action_dict, alive_worms, self.flood_list,
                self.terrain, self.attachment_system, self.worms_by_id, boost
            )

            action_results[worm_id] = success
            actions_dict[worm_id] = action_vector
            log_probs_dict[worm_id] = float(log_probs_np[i])
            values_dict[worm_id] = float(values_np[i])
            worm_type_dict[worm_id] = int(worm.worm_type)
            dones_dict[worm_id] = not worm.alive

        # --- Phase 4: Energy regeneration ---
        for worm in alive_worms:
            worm.regenerate_energy(WORM_ENERGY_REGEN)

        # --- Phase 5: Flood actions ---
        prev_flood_count = sum(1 for f in self.flood_list if f.alive)
        flood_biomass_gained = 0.0
        new_flood = []

        for flood in self.flood_list:
            if not flood.alive:
                continue

            flood.tick_timers()

            # Find nearby worms for Flood AI
            nearby_worms = []
            for worm in alive_worms:
                d = flood.distance_to(worm.x, worm.y)
                if d <= FLOOD_VISION:
                    nearby_worms.append(worm)

            # Find nearby flood for coordination
            nearby_flood = []
            for other in self.flood_list:
                if other.id == flood.id or not other.alive:
                    continue
                d = flood.distance_to(other.x, other.y)
                if d <= FLOOD_VISION:
                    nearby_flood.append(other)

            # Get flood action
            action = flood.choose_action(nearby_worms, nearby_flood, self.terrain)

            if action['type'] == 'move':
                new_x = flood.x + action['dx']
                new_y = flood.y + action['dy']
                # Check bounds and terrain
                gx = int(np.clip(np.round(new_x), 0, self.width - 1))
                gy = int(np.clip(np.round(new_y), 0, self.height - 1))
                if is_passable(self.terrain, gx, gy):
                    flood.x = np.clip(new_x, 0, self.width - 1)
                    flood.y = np.clip(new_y, 0, self.height - 1)

            elif action['type'] == 'attack':
                target_id = action.get('target_id')
                if target_id is not None and target_id in self.worms_by_id:
                    target = self.worms_by_id[target_id]
                    if target.alive:
                        d = flood.distance_to(target.x, target.y)
                        if d <= FLOOD_ATTACK_RANGE:
                            # Flood deals damage
                            damage = FLOOD_ATTACK_DAMAGE
                            # Apply structural damage reduction
                            reduction = self.attachment_system.compute_damage_reduction(target_id)
                            actual_damage = damage * (1.0 - reduction)
                            blocked = damage * reduction
                            target.take_damage(actual_damage)
                            target.damage_blocked_this_step += blocked

                            # Infection chance
                            if np.random.random() < 0.3:  # 30% infection chance per hit
                                target.infected = True
                                target.infection_timer += 1
                                infection_reward = self.flood_reward_tracker.compute_infection_reward(
                                    int(target.worm_type)
                                )
                                # Apply infection damage over time
                                target.take_damage(FLOOD_INFECTION_DAMAGE * 0.1)

            # Flood reproduction
            if action.get('reproduce', False) and flood.can_reproduce():
                if len(self.flood_list) + len(new_flood) < MAX_FLOOD_COUNT:
                    offset_x = self.rng.normal(0, 2)
                    offset_y = self.rng.normal(0, 2)
                    child = FloodOrganism(
                        np.clip(flood.x + offset_x, 0, self.width - 1),
                        np.clip(flood.y + offset_y, 0, self.height - 1)
                    )
                    new_flood.append(child)
                    flood.biomass -= FLOOD_REPRODUCE_BIOMASS_COST
                    flood.reproduce_timer = FLOOD_REPRODUCE_INTERVAL
                    flood_biomass_gained += 1.0

        # Add new flood
        self.flood_list.extend(new_flood)
        for f in new_flood:
            self.flood_by_id[f.id] = f

        # --- Phase 6: Infection damage over time ---
        for worm in alive_worms:
            if worm.infected and worm.alive:
                worm.infection_timer += 1
                # Infection spreads damage over time
                worm.take_damage(FLOOD_INFECTION_DAMAGE * 0.05)
                # Small chance to recover
                if np.random.random() < 0.01:
                    worm.infected = False
                    worm.infection_timer = 0

        # --- Phase 7: Clean up dead entities ---
        dead_worms = [w for w in self.worms if not w.alive]
        for worm in dead_worms:
            # Remove attachments
            for nid in list(worm.attachments):
                self.attachment_system.remove_edge(worm.id, nid)
                if nid in self.worms_by_id:
                    self.worms_by_id[nid].attachments.discard(worm.id)
            worm.attachments.clear()

        self.flood_list = [f for f in self.flood_list if f.alive]

        # --- Phase 8: Spawn new Flood at edges ---
        for _ in range(FLOOD_SPAWN_RATE):
            self._spawn_flood_at_edge()

        # --- Phase 9: Biomass accumulation ---
        flood_kills = sum(w.flood_kills_this_step for w in alive_worms)
        self.biomass += flood_kills * BIOMASS_PER_FLOOD_KILL
        alive_count = sum(1 for w in self.worms if w.alive)
        self.biomass += alive_count * BIOMASS_PER_TIMESTEP

        # --- Phase 10: Compute rewards ---
        alive_worm_ids = {w.id for w in self.worms if w.alive}
        colony_reward = self.colony_reward_tracker.compute_step_reward(
            self.worms, self.flood_list, self.attachment_system,
            self.terrain, self.biomass
        )

        # Check colony fragmentation
        prev_frags = self.colony_reward_tracker.prev_fragment_count
        current_frags = self.attachment_system.count_fragments(alive_worm_ids)
        colony_fragmented = current_frags > prev_frags

        flood_reward = self.flood_reward_tracker.compute_step_reward(
            self.flood_list, prev_flood_count, colony_fragmented,
            flood_biomass_gained
        )

        self.episode_reward += colony_reward

        # --- Phase 11: Collect rollout data ---
        self.trainer.collect_step(
            observations, actions_dict, log_probs_dict, values_dict,
            colony_reward, dones_dict, worm_type_dict
        )

        # --- Phase 12: Reset per-step trackers ---
        for worm in alive_worms:
            worm.damage_blocked_this_step = 0.0
            worm.flood_kills_this_step = 0
            worm.local_damage_taken = 0.0

        # --- Phase 13: Age attachments ---
        self.attachment_system.tick_edges()

        # --- Check done ---
        done = (
            self.timestep >= MAX_STEPS_PER_EPISODE or
            not any(w.alive for w in self.worms)
        )

        # Build new observations for next step
        new_observations = self._build_all_observations()

        info = {
            'timestep': self.timestep,
            'alive_worms': sum(1 for w in self.worms if w.alive),
            'alive_thinkers': sum(1 for w in self.worms if w.alive and w.worm_type == WormType.THINKER),
            'alive_workers': sum(1 for w in self.worms if w.alive and w.worm_type == WormType.WORKER),
            'alive_flood': len(self.flood_list),
            'biomass': self.biomass,
            'colony_fragments': current_frags,
            'attachments': len(self.attachment_system.edges),
            'infected_worms': sum(1 for w in self.worms if w.alive and w.infected),
            'episode_reward': self.episode_reward,
            'colony_reward_breakdown': self.colony_reward_tracker.reward_breakdown,
        }

        return {
            'observations': new_observations,
            'colony_reward': colony_reward,
            'flood_reward': flood_reward,
            'done': done,
            'info': info,
        }

    def run_episode(self, max_steps: int | None = None,
                    render: bool = False,
                    train: bool = True) -> dict:
        """
        Run a full episode.

        Args:
            max_steps: override for max steps per episode
            render: whether to render during the episode
            train: whether to perform PPO updates during the episode

        Returns:
            Episode statistics dict
        """
        max_steps = max_steps or MAX_STEPS_PER_EPISODE
        observations = self.reset()

        episode_stats = []
        update_interval = PPO_ROLLOUT_LENGTH

        for step in range(max_steps):
            result = self.step()
            observations = result['observations']

            if render and step % 10 == 0:
                self.render(save_path=f'/home/z/my-project/download/frame_{step:06d}.png')

            # Periodic PPO update
            if train and len(self.trainer.buffer) >= update_interval:
                next_value = 0.0
                if result['info'].get('alive_worms', 0) > 0:
                    # Estimate next value from current observations
                    with torch.no_grad():
                        obs_list = list(observations.values())
                        if obs_list:
                            obs_batch = torch.FloatTensor(np.array(obs_list)).to(self.device)
                            types = [int(self.worms_by_id[wid].worm_type)
                                     for wid in observations
                                     if wid in self.worms_by_id]
                            if types:
                                type_batch = torch.LongTensor(types).to(self.device)
                                _, _, vals, _ = self.policy.get_action(obs_batch, type_batch)
                                next_value = vals.mean().item()

                stats = self.trainer.update(next_value)
                episode_stats.append(stats)

            if result['done']:
                break

        # Final update with remaining buffer
        if train and len(self.trainer.buffer) > 0:
            stats = self.trainer.update(0.0)
            episode_stats.append(stats)

        return {
            'episode_reward': self.episode_reward,
            'timesteps': self.timestep,
            'training_stats': episode_stats,
            'final_info': result.get('info', {}),
            'colony_reward_breakdown': self.colony_reward_tracker.reward_breakdown,
        }

    def render(self, save_path: str | None = None, show: bool = False):
        """
        Render the current state of the simulation.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
        fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        plt.rcParams['font.sans-serif'] = ['Sarasa Mono SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Draw terrain
        terrain_colors = {0: [0.2, 0.5, 0.2], 1: [0.5, 0.4, 0.2], 2: [0.3, 0.3, 0.3]}
        terrain_img = np.zeros((self.height, self.width, 3))
        for t_val, color in terrain_colors.items():
            terrain_img[self.terrain == t_val] = color
        ax.imshow(terrain_img, extent=[0, self.width, 0, self.height], alpha=0.6)

        # Draw attachments
        alive_worms = {w.id: w for w in self.worms if w.alive}
        for (a, b) in self.attachment_system.edges:
            if a in alive_worms and b in alive_worms:
                wa, wb = alive_worms[a], alive_worms[b]
                ax.plot([wa.x, wb.x], [wa.y, wb.y], 'y-', alpha=0.3, linewidth=0.5)

        # Draw workers
        workers = [w for w in self.worms if w.alive and w.worm_type == WormType.WORKER]
        if workers:
            wx = [w.x for w in workers]
            wy = [w.y for w in workers]
            wh = [w.health / w.max_health for w in workers]
            infected = [w.infected for w in workers]
            colors = ['red' if inf else 'lime' for inf in infected]
            ax.scatter(wx, wy, c=colors, s=8, alpha=0.7, zorder=3)

        # Draw thinkers
        thinkers = [w for w in self.worms if w.alive and w.worm_type == WormType.THINKER]
        if thinkers:
            tx = [w.x for w in thinkers]
            ty = [w.y for w in thinkers]
            infected_t = [w.infected for w in thinkers]
            colors_t = ['red' if inf else 'cyan' for inf in infected_t]
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

        info_text = (
            f"Step: {self.timestep}  |  "
            f"Worms: {alive_w} (Thinkers: {alive_t})  |  "
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
        """Save full environment state and policy."""
        self.trainer.save(path)

    def load_checkpoint(self, path: str):
        """Load policy from checkpoint."""
        self.trainer.load(path)
