"""
Training script for the Lekgolo Colony simulation.

Runs PPO training across multiple episodes and logs progress.
"""
import sys
import os
import time
import json
import numpy as np

# Add the lekgolo directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import LekgoloEnvironment
from config import MAX_STEPS_PER_EPISODE


def train(num_episodes: int = 1000,
          save_interval: int = 10,
          render_interval: int = 50,
          checkpoint_dir: str = '/home/z/my-project/download/checkpoints'):
    """
    Train the Lekgolo colony using PPO.

    Args:
        num_episodes: number of episodes to train
        save_interval: save checkpoint every N episodes
        render_interval: render and save frame every N episodes
        checkpoint_dir: directory for checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = LekgoloEnvironment(seed=42)
    all_stats = []

    print("=" * 70)
    print("LEKGLO COLONY SIMULATION - PPO TRAINING")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 70)

    for episode in range(num_episodes):
        start_time = time.time()

        # Run episode
        do_render = (episode % render_interval == 0)
        result = env.run_episode(
            max_steps=MAX_STEPS_PER_EPISODE,
            render=do_render,
            train=True,
        )

        elapsed = time.time() - start_time

        # Extract stats
        final_info = result.get('final_info', {})
        reward_breakdown = result.get('colony_reward_breakdown', {})
        training_stats = result.get('training_stats', [])

        episode_stats = {
            'episode': episode,
            'reward': result['episode_reward'],
            'timesteps': result['timesteps'],
            'elapsed_sec': elapsed,
            'final_info': final_info,
        }
        all_stats.append(episode_stats)

        # Print progress
        alive_worms = final_info.get('alive_worms', 0)
        alive_thinkers = final_info.get('alive_thinkers', 0)
        alive_flood = final_info.get('alive_flood', 0)
        attachments = final_info.get('attachments', 0)
        fragments = final_info.get('colony_fragments', 0)
        infected = final_info.get('infected_worms', 0)

        # Average training loss
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        if training_stats:
            avg_policy_loss = np.mean([s.get('policy_loss', 0) for s in training_stats])
            avg_value_loss = np.mean([s.get('value_loss', 0) for s in training_stats])
            avg_entropy = np.mean([s.get('entropy', 0) for s in training_stats])

        print(f"\nEpisode {episode:4d} | "
              f"Reward: {result['episode_reward']:8.2f} | "
              f"Steps: {result['timesteps']:4d} | "
              f"Alive: {alive_worms:3d} (T:{alive_thinkers:2d}) | "
              f"Flood: {alive_flood:3d} | "
              f"Links: {attachments:4d} | "
              f"Frags: {fragments:2d} | "
              f"Infected: {infected:3d} | "
              f"P_loss: {avg_policy_loss:6.4f} | "
              f"V_loss: {avg_value_loss:6.4f} | "
              f"Entropy: {avg_entropy:6.4f} | "
              f"Time: {elapsed:5.1f}s")

        # Print reward breakdown every 10 episodes
        if episode % 10 == 0 and reward_breakdown:
            print(f"  Reward breakdown:")
            for key, val in reward_breakdown.items():
                print(f"    {key}: {val:8.4f}")

        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            ckpt_path = os.path.join(checkpoint_dir, f'episode_{episode:05d}.pt')
            env.save_checkpoint(ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            # Also save a render
            render_path = os.path.join(checkpoint_dir, f'episode_{episode:05d}.png')
            env.render(save_path=render_path)
            print(f"  Saved render: {render_path}")

        # Save stats periodically
        if episode % 10 == 0:
            stats_path = os.path.join(checkpoint_dir, 'training_stats.json')
            with open(stats_path, 'w') as f:
                # Convert numpy types for JSON serialization
                serializable_stats = []
                for s in all_stats:
                    clean = {}
                    for k, v in s.items():
                        if isinstance(v, (np.integer,)):
                            clean[k] = int(v)
                        elif isinstance(v, (np.floating,)):
                            clean[k] = float(v)
                        elif isinstance(v, dict):
                            clean[k] = {
                                kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                for kk, vv in v.items()
                            }
                        else:
                            clean[k] = v
                    serializable_stats.append(clean)
                json.dump(serializable_stats, f, indent=2)

    # Final save
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    env.save_checkpoint(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    return all_stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Lekgolo Colony')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--render-interval', type=int, default=50,
                        help='Render every N episodes')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/home/z/my-project/download/checkpoints',
                        help='Checkpoint directory')
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        save_interval=args.save_interval,
        render_interval=args.render_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
