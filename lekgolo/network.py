"""
Neural network architectures for Lekgolo worm policies.

Worker: 16 hidden units
Thinker: 128 hidden units
Same architecture, different capacity.

The colony's physical structure (attachments) determines information flow.
Attached worms share observations through the graph structure.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from config import (
    WORKER_HIDDEN_DIM, THINKER_HIDDEN_DIM, SIGNAL_DIM,
    NUM_DISCRETE_ACTIONS, ACTION_PARAM_DIM
)


class WormPolicyNetwork(nn.Module):
    """
    Policy network for a single worm.

    Architecture:
        Observation -> MLP -> action_logits (discrete) + action_params (continuous)

    The hidden dimension differs between workers and thinkers:
    - Workers: 16 hidden units (small, cheap, many)
    - Thinkers: 128 hidden units (large, expensive, few)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = WORKER_HIDDEN_DIM):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Discrete action head (6 actions)
        self.action_head = nn.Linear(hidden_dim, NUM_DISCRETE_ACTIONS)

        # Continuous parameter head
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_PARAM_DIM),
        )

        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Log standard deviation for continuous actions
        self.log_std = nn.Parameter(torch.zeros(ACTION_PARAM_DIM))

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Smaller gain for policy heads
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.orthogonal_(self.param_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim) tensor

        Returns:
            action_logits: (batch, NUM_DISCRETE_ACTIONS)
            action_params_mean: (batch, ACTION_PARAM_DIM)
            action_params_std: (batch, ACTION_PARAM_DIM)
            value: (batch, 1)
        """
        features = self.feature_net(obs)
        action_logits = self.action_head(features)
        action_params_mean = torch.tanh(self.param_head(features))  # bounded [-1, 1]
        action_params_std = torch.exp(self.log_std).expand_as(action_params_mean)
        value = self.value_head(features)

        return action_logits, action_params_mean, action_params_std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample an action from the policy.

        Returns:
            action: (batch, 1 + ACTION_PARAM_DIM) - discrete choice + continuous params
            log_prob: (batch,)
            value: (batch, 1)
            entropy: (batch,)
        """
        action_logits, param_mean, param_std, value = self.forward(obs)

        # Discrete action
        dist_discrete = Categorical(logits=action_logits)
        if deterministic:
            discrete_action = action_logits.argmax(dim=-1)
        else:
            discrete_action = dist_discrete.sample()

        # Continuous parameters
        dist_continuous = Normal(param_mean, param_std)
        if deterministic:
            continuous_params = param_mean
        else:
            continuous_params = dist_continuous.rsample()
        continuous_params = torch.clamp(continuous_params, -1.0, 1.0)

        # Combine into action vector
        action = torch.cat([
            discrete_action.float().unsqueeze(-1),
            continuous_params
        ], dim=-1)

        # Compute log probability
        log_prob_discrete = dist_discrete.log_prob(discrete_action)
        log_prob_continuous = dist_continuous.log_prob(continuous_params).sum(dim=-1)
        log_prob = log_prob_discrete + log_prob_continuous

        # Entropy for exploration bonus
        entropy = dist_discrete.entropy() + dist_continuous.entropy().sum(dim=-1)

        return action, log_prob, value, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate log probabilities and values for given actions.
        Used during PPO update.

        Args:
            obs: (batch, obs_dim)
            actions: (batch, 1 + ACTION_PARAM_DIM)

        Returns:
            log_prob: (batch,)
            value: (batch, 1)
            entropy: (batch,)
        """
        action_logits, param_mean, param_std, value = self.forward(obs)

        discrete_actions = actions[:, 0].long()
        continuous_params = actions[:, 1:]

        dist_discrete = Categorical(logits=action_logits)
        dist_continuous = Normal(param_mean, param_std)

        log_prob_discrete = dist_discrete.log_prob(discrete_actions)
        log_prob_continuous = dist_continuous.log_prob(continuous_params).sum(dim=-1)
        log_prob = log_prob_discrete + log_prob_continuous

        entropy = dist_discrete.entropy() + dist_continuous.entropy().sum(dim=-1)

        return log_prob, value, entropy


class ColonySharedPolicy(nn.Module):
    """
    Shared policy that uses the same network for all worms of the same type.

    Workers share one network (16 hidden).
    Thinkers share one network (128 hidden).

    This is parameter-efficient and allows knowledge sharing
    across worms of the same type.
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.worker_policy = WormPolicyNetwork(obs_dim, WORKER_HIDDEN_DIM)
        self.thinker_policy = WormPolicyNetwork(obs_dim, THINKER_HIDDEN_DIM)

    def get_policy(self, worm_type: int) -> WormPolicyNetwork:
        if worm_type == 1:  # Thinker
            return self.thinker_policy
        return self.worker_policy

    def get_action(self, obs: torch.Tensor, worm_type: torch.Tensor,
                   deterministic: bool = False):
        """
        Get actions for a batch of mixed worm types.

        obs: (batch, obs_dim)
        worm_type: (batch,) - 0 for worker, 1 for thinker
        """
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, 1 + ACTION_PARAM_DIM, device=obs.device)
        log_probs = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, 1, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)

        # Process workers
        worker_mask = (worm_type == 0)
        if worker_mask.any():
            worker_obs = obs[worker_mask]
            a, lp, v, e = self.worker_policy.get_action(worker_obs, deterministic)
            actions[worker_mask] = a
            log_probs[worker_mask] = lp
            values[worker_mask] = v
            entropies[worker_mask] = e

        # Process thinkers
        thinker_mask = (worm_type == 1)
        if thinker_mask.any():
            thinker_obs = obs[thinker_mask]
            a, lp, v, e = self.thinker_policy.get_action(thinker_obs, deterministic)
            actions[thinker_mask] = a
            log_probs[thinker_mask] = lp
            values[thinker_mask] = v
            entropies[thinker_mask] = e

        return actions, log_probs, values, entropies

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         worm_type: torch.Tensor):
        """Evaluate actions for PPO update."""
        batch_size = obs.shape[0]
        log_probs = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, 1, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)

        worker_mask = (worm_type == 0)
        if worker_mask.any():
            lp, v, e = self.worker_policy.evaluate_actions(
                obs[worker_mask], actions[worker_mask]
            )
            log_probs[worker_mask] = lp
            values[worker_mask] = v
            entropies[worker_mask] = e

        thinker_mask = (worm_type == 1)
        if thinker_mask.any():
            lp, v, e = self.thinker_policy.evaluate_actions(
                obs[thinker_mask], actions[thinker_mask]
            )
            log_probs[thinker_mask] = lp
            values[thinker_mask] = v
            entropies[thinker_mask] = e

        return log_probs, values, entropies

    def get_all_parameters(self) -> list:
        """Get all trainable parameters."""
        return list(self.worker_policy.parameters()) + list(self.thinker_policy.parameters())
