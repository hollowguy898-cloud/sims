"""
PPO (Proximal Policy Optimization) trainer for the Lekgolo colony.

Uses the colony shared policy with separate worker/thinker networks.
Implements the standard PPO algorithm with GAE (Generalized Advantage Estimation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import (
    PPO_LEARNING_RATE, PPO_GAMMA, PPO_GAE_LAMBDA,
    PPO_CLIP_EPSILON, PPO_ENTROPY_COEFF, PPO_VALUE_LOSS_COEFF,
    PPO_MAX_GRAD_NORM, PPO_EPOCHS, PPO_MINIBATCH_SIZE,
    PPO_ROLLOUT_LENGTH
)


class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.worm_types = []

    def add(self, obs: np.ndarray, action: np.ndarray, log_prob: float,
            reward: float, value: float, done: bool, worm_type: int):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.worm_types.append(worm_type)

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.worm_types = []

    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO trainer for the colony shared policy.

    Collects rollouts from all worms, computes advantages with GAE,
    and performs multiple epochs of minibatch updates.
    """

    def __init__(self, policy, device: str = 'cpu'):
        self.policy = policy
        self.device = device
        self.optimizer = optim.Adam(
            policy.get_all_parameters(),
            lr=PPO_LEARNING_RATE,
            eps=1e-5
        )
        self.buffer = RolloutBuffer()
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
        }

    def collect_step(self, obs_dict: dict, actions_dict: dict,
                     log_probs_dict: dict, values_dict: dict,
                     reward: float, dones_dict: dict, worm_type_dict: dict):
        """
        Collect one step of rollout data from all worms.

        Args:
            obs_dict: {worm_id: observation_array}
            actions_dict: {worm_id: action_array}
            log_probs_dict: {worm_id: log_prob_float}
            values_dict: {worm_id: value_float}
            reward: colony-level reward for this step
            dones_dict: {worm_id: bool}
            worm_type_dict: {worm_id: WormType}
        """
        for wid in obs_dict:
            self.buffer.add(
                obs=obs_dict[wid],
                action=actions_dict[wid],
                log_prob=log_probs_dict.get(wid, 0.0),
                reward=reward,  # shared colony reward
                value=values_dict.get(wid, 0.0),
                done=dones_dict.get(wid, False),
                worm_type=worm_type_dict[wid],
            )

    def compute_gae(self, next_value: float = 0.0) -> tuple:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            advantages: numpy array
            returns: numpy array (advantages + values)
        """
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        values = np.array(self.buffer.values, dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + PPO_GAMMA * next_val * next_non_terminal - values[t]
            last_gae = delta + PPO_GAMMA * PPO_GAE_LAMBDA * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_value: float = 0.0) -> dict:
        """
        Perform a PPO update using collected rollout data.

        Returns:
            stats: dict of training statistics
        """
        if len(self.buffer) < PPO_MINIBATCH_SIZE:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0,
                    'total_loss': 0, 'approx_kl': 0}

        # Compute advantages
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        obs_t = torch.FloatTensor(np.array(self.buffer.observations)).to(self.device)
        actions_t = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(-1)
        worm_types_t = torch.LongTensor(np.array(self.buffer.worm_types)).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        total_kl = 0
        num_updates = 0

        dataset = TensorDataset(obs_t, actions_t, old_log_probs_t,
                                advantages_t, returns_t, worm_types_t)
        dataloader = DataLoader(dataset, batch_size=PPO_MINIBATCH_SIZE,
                                shuffle=True)

        for epoch in range(PPO_EPOCHS):
            for batch in dataloader:
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns, b_worm_types = batch

                # Evaluate current policy
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    b_obs, b_actions, b_worm_types
                )

                # Compute ratio for PPO clip
                ratio = torch.exp(new_log_probs - b_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON,
                                    1.0 + PPO_CLIP_EPSILON) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = F.mse_loss(new_values, b_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        PPO_VALUE_LOSS_COEFF * value_loss +
                        PPO_ENTROPY_COEFF * entropy_loss)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.get_all_parameters(),
                                        PPO_MAX_GRAD_NORM)
                self.optimizer.step()

                # Track stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                total_kl += (ratio - 1.0).abs().mean().item()
                num_updates += 1

        # Compute averages
        stats = {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
            'total_loss': total_loss / max(num_updates, 1),
            'approx_kl': total_kl / max(num_updates, 1),
        }

        for key in stats:
            self.training_stats[key].append(stats[key])

        # Clear buffer
        self.buffer.clear()

        return stats

    def save(self, path: str):
        """Save policy and optimizer state."""
        torch.save({
            'worker_policy': self.policy.worker_policy.state_dict(),
            'thinker_policy': self.policy.thinker_policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
        }, path)

    def load(self, path: str):
        """Load policy and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.worker_policy.load_state_dict(checkpoint['worker_policy'])
        self.policy.thinker_policy.load_state_dict(checkpoint['thinker_policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
