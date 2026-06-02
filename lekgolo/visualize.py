"""
Standalone visualization and analysis tool for Lekgolo Colony simulations.

Can be used to:
- Visualize a saved checkpoint
- Run a simulation with real-time rendering
- Analyze training statistics
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import LekgoloEnvironment
from config import MAX_STEPS_PER_EPISODE, MAP_WIDTH, MAP_HEIGHT, setup_matplotlib_fonts, DEFAULT_FRAME_DIR, DEFAULT_CHECKPOINT_DIR


def render_episode(checkpoint_path: str | None = None,
                   num_steps: int = 500,
                   output_dir: str | None = None,
                   seed: int = 42):
    """
    Run and render an episode, saving frames as PNGs.

    Args:
        checkpoint_path: path to saved model checkpoint (or None for random)
        num_steps: number of steps to simulate
        output_dir: directory for frame images
        seed: random seed
    """
    output_dir = output_dir or DEFAULT_FRAME_DIR
    os.makedirs(output_dir, exist_ok=True)

    env = LekgoloEnvironment(seed=seed)

    if checkpoint_path and os.path.exists(checkpoint_path):
        env.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}")

    observations = env.reset()

    print(f"Running {num_steps} steps...")
    for step in range(num_steps):
        # Render every 5 steps
        if step % 5 == 0:
            frame_path = os.path.join(output_dir, f'frame_{step:06d}.png')
            env.render(save_path=frame_path)

        result = env.step()
        observations = result['observations']

        if step % 50 == 0:
            info = result.get('info', {})
            print(f"  Step {step}: "
                  f"Alive={info.get('alive_worms', 0)} "
                  f"Thinkers={info.get('alive_thinkers', 0)} "
                  f"Flood={info.get('alive_flood', 0)} "
                  f"Links={info.get('attachments', 0)} "
                  f"Frags={info.get('colony_fragments', 0)}")

        if result['done']:
            print(f"Episode ended at step {step}")
            break

    print(f"Frames saved to: {output_dir}")


def analyze_training(stats_path: str,
                     output_path: str | None = None):
    """
    Analyze and plot training statistics from a training_stats.json file.

    Args:
        stats_path: path to training_stats.json
        output_path: path for the output plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = output_path or os.path.join(DEFAULT_FRAME_DIR, 'training_analysis.png')
    setup_matplotlib_fonts()

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    if not stats:
        print("No stats to analyze.")
        return

    episodes = [s['episode'] for s in stats]
    rewards = [s['reward'] for s in stats]
    timesteps = [s['timesteps'] for s in stats]
    alive_worms = [s.get('final_info', {}).get('alive_worms', 0) for s in stats]
    alive_thinkers = [s.get('final_info', {}).get('alive_thinkers', 0) for s in stats]
    attachments = [s.get('final_info', {}).get('attachments', 0) for s in stats]
    fragments = [s.get('final_info', {}).get('colony_fragments', 0) for s in stats]
    infected = [s.get('final_info', {}).get('infected_worms', 0) for s in stats]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Episode rewards
    axes[0, 0].plot(episodes, rewards, alpha=0.5, color='blue')
    # Smoothed
    if len(rewards) > 10:
        window = min(20, len(rewards))
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(episodes[:len(smoothed)], smoothed, color='red', linewidth=2)
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')

    # Alive worms over episodes
    axes[0, 1].plot(episodes, alive_worms, label='Total Worms', alpha=0.7)
    axes[0, 1].plot(episodes, alive_thinkers, label='Thinkers', alpha=0.7)
    axes[0, 1].set_title('Survivors')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend(loc='best')

    # Attachments
    axes[1, 0].plot(episodes, attachments, color='orange', alpha=0.7)
    axes[1, 0].set_title('Colony Attachments (Links)')
    axes[1, 0].set_xlabel('Episode')

    # Colony fragments
    axes[1, 1].plot(episodes, fragments, color='red', alpha=0.7)
    axes[1, 1].set_title('Colony Fragments')
    axes[1, 1].set_xlabel('Episode')

    # Infected worms
    axes[2, 0].plot(episodes, infected, color='darkred', alpha=0.7)
    axes[2, 0].set_title('Infected Worms')
    axes[2, 0].set_xlabel('Episode')

    # Timesteps survived
    axes[2, 1].plot(episodes, timesteps, color='green', alpha=0.7)
    axes[2, 1].set_title('Timesteps Survived')
    axes[2, 1].set_xlabel('Episode')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Analysis saved to: {output_path}")


def create_single_frame(output_path: str | None = None):
    """Create a single visualization frame of the current simulation state."""
    output_path = output_path or os.path.join(DEFAULT_FRAME_DIR, 'lekgolo_state.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = LekgoloEnvironment(seed=123)
    env.reset()

    # Run a few steps to get interesting state
    for _ in range(50):
        result = env.step()
        if result['done']:
            break

    env.render(save_path=output_path)
    print(f"Frame saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Lekgolo Visualization')
    parser.add_argument('--mode', choices=['render', 'analyze', 'snapshot'],
                        default='snapshot', help='Visualization mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint')
    parser.add_argument('--steps', type=int, default=500,
                        help='Number of steps to render')
    parser.add_argument('--stats', type=str, default=None,
                        help='Path to training_stats.json')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Output directory')
    args = parser.parse_args()

    if args.mode == 'render':
        render_episode(args.checkpoint, args.steps, args.output_dir or DEFAULT_FRAME_DIR)
    elif args.mode == 'analyze':
        stats_path = args.stats or os.path.join(DEFAULT_CHECKPOINT_DIR, 'training_stats.json')
        analyze_training(stats_path)
    elif args.mode == 'snapshot':
        create_single_frame()
