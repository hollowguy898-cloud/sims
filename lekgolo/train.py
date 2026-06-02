"""
Training script for the Lekgolo Colony simulation.

Runs PPO training for BOTH sides:
  - Lekgolo colony (workers + thinkers)
  - Flood (replication agents)

Both sides learn. Both sides adapt. Both sides will absolutely
find ways to break your reward function in ways you did not
emotionally consent to.
"""
import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import LekgoloEnvironment
from config import MAX_STEPS_PER_EPISODE, DEFAULT_CHECKPOINT_DIR


def train(num_episodes: int = 1000,
          save_interval: int = 10,
          render_interval: int = 50,
          checkpoint_dir: str | None = None):
    """Train both Lekgolo and Flood policies using PPO."""
    checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = LekgoloEnvironment(seed=42)
    all_stats = []

    print("=" * 70)
    print("LEKGLO COLONY vs FLOOD - ASYMMETRIC PPO TRAINING")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 70)
    print("Lekgolo: structure + coordination + thinker protection")
    print("Flood:   replication + disruption + spread")
    print("Neither side is scripted. Both evolve.")
    print("=" * 70)

    for episode in range(num_episodes):
        start_time = time.time()

        do_render = (episode % render_interval == 0)
        result = env.run_episode(
            max_steps=MAX_STEPS_PER_EPISODE,
            render=do_render,
            train=True,
        )

        elapsed = time.time() - start_time
        final_info = result.get('final_info', {})
        training_stats = result.get('training_stats', [])

        episode_stats = {
            'episode': episode,
            'lekgolo_reward': result['episode_reward'],
            'flood_reward': result.get('flood_episode_reward', 0),
            'timesteps': result['timesteps'],
            'elapsed_sec': elapsed,
            'map_type': result.get('map_type', 'unknown'),
        }
        all_stats.append(episode_stats)

        alive_worms = final_info.get('alive_worms', 0)
        alive_thinkers = final_info.get('alive_thinkers', 0)
        alive_flood = final_info.get('alive_flood', 0)
        attachments = final_info.get('attachments', 0)
        fragments = final_info.get('colony_fragments', 0)
        infected = final_info.get('infected_worms', 0)

        print(f"\nEp {episode:4d} [{result.get('map_type', '?'):6s}] | "
              f"Lekgolo: {result['episode_reward']:8.1f} | "
              f"Flood: {result.get('flood_episode_reward', 0):8.1f} | "
              f"Steps: {result['timesteps']:4d} | "
              f"Alive: {alive_worms:3d} (T:{alive_thinkers:2d}) | "
              f"Flood: {alive_flood:3d} | "
              f"Links: {attachments:4d} | "
              f"Inf: {infected:3d} | "
              f"Time: {elapsed:5.1f}s")

        if episode % 10 == 0:
            lekgolo_bd = result.get('colony_reward_breakdown', {})
            flood_bd = result.get('flood_reward_breakdown', {})
            if lekgolo_bd:
                print(f"  Lekgolo rewards: {', '.join(f'{k}={v:.2f}' for k, v in lekgolo_bd.items() if v != 0)}")
            if flood_bd:
                print(f"  Flood rewards:   {', '.join(f'{k}={v:.2f}' for k, v in flood_bd.items() if v != 0)}")

        if episode % save_interval == 0 and episode > 0:
            ckpt_path = os.path.join(checkpoint_dir, f'episode_{episode:05d}.pt')
            env.save_checkpoint(ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

            render_path = os.path.join(checkpoint_dir, f'episode_{episode:05d}.png')
            env.render(save_path=render_path)

        if episode % 10 == 0:
            stats_path = os.path.join(checkpoint_dir, 'training_stats.json')
            with open(stats_path, 'w') as f:
                serializable = []
                for s in all_stats:
                    clean = {}
                    for k, v in s.items():
                        if isinstance(v, (np.integer,)):
                            clean[k] = int(v)
                        elif isinstance(v, (np.floating,)):
                            clean[k] = float(v)
                        else:
                            clean[k] = v
                    serializable.append(clean)
                json.dump(serializable, f, indent=2)

    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    env.save_checkpoint(final_path)
    print(f"\nTraining complete. Final model: {final_path}")
    return all_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Lekgolo vs Flood')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--render-interval', type=int, default=50)
    parser.add_argument('--checkpoint-dir', type=str,
                        default=None)
    args = parser.parse_args()
    train(num_episodes=args.episodes, save_interval=args.save_interval,
          render_interval=args.render_interval, checkpoint_dir=args.checkpoint_dir)
