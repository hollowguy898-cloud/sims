"""
Neural network architectures for Lekgolo worm and Flood policies.

Lekgolo:
  Worker: 16 hidden units
  Thinker: 128 hidden units
  Same architecture, different capacity.

Flood:
  32 hidden units (simpler - high birth rate, low individual intelligence)

The colony's physical structure (attachments) determines information flow.
Attached worms share observations through the graph structure.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from config import (
    WORKER_HIDDEN_DIM, THINKER_HIDDEN_DIM, FLOOD_HIDDEN_DIM,
    SIGNAL_DIM, NUM_DISCRETE_ACTIONS, ACTION_PARAM_DIM,
    FLOOD_NUM_DISCRETE_ACTIONS, FLOOD_ACTION_PARAM_DIM,
)


class WormPolicyNetwork(nn.Module):
    """
    Policy network for a single Lekgolo worm.

    Architecture:
        Observation -> MLP -> action_logits (discrete) + action_params (continuous)

    The hidden dimension differs between workers and thinkers:
    - Workers: 16 hidden units (small, cheap, many)
    - Thinkers: 128 hidden units (large, expensive, few)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = WORKER_HIDDEN_DIM,
                 num_actions: int = NUM_DISCRETE_ACTIONS,
                 param_dim: int = ACTION_PARAM_DIM):
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

        # Discrete action head
        self.action_head = nn.Linear(hidden_dim, num_actions)

        # Continuous parameter head
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
        )

        # Value head for PPO
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Log standard deviation for continuous actions
        self.log_std = nn.Parameter(torch.zeros(param_dim))

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.orthogonal_(self.param_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        features = self.feature_net(obs)
        action_logits = self.action_head(features)
        action_params_mean = torch.tanh(self.param_head(features))
        action_params_std = torch.exp(self.log_std).expand_as(action_params_mean)
        value = self.value_head(features)
        return action_logits, action_params_mean, action_params_std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        action_logits, param_mean, param_std, value = self.forward(obs)

        dist_discrete = Categorical(logits=action_logits)
        if deterministic:
            discrete_action = action_logits.argmax(dim=-1)
        else:
            discrete_action = dist_discrete.sample()

        dist_continuous = Normal(param_mean, param_std)
        if deterministic:
            continuous_params = param_mean
        else:
            continuous_params = dist_continuous.rsample()
        continuous_params = torch.clamp(continuous_params, -1.0, 1.0)

        action = torch.cat([
            discrete_action.float().unsqueeze(-1),
            continuous_params
        ], dim=-1)

        log_prob_discrete = dist_discrete.log_prob(discrete_action)
        log_prob_continuous = dist_continuous.log_prob(continuous_params).sum(dim=-1)
        log_prob = log_prob_discrete + log_prob_continuous

        entropy = dist_discrete.entropy() + dist_continuous.entropy().sum(dim=-1)

        return action, log_prob, value, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
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
    Shared policy for all Lekgolo worms of the same type.
    Workers share one network (16 hidden).
    Thinkers share one network (128 hidden).
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.worker_policy = WormPolicyNetwork(obs_dim, WORKER_HIDDEN_DIM,
                                               NUM_DISCRETE_ACTIONS, ACTION_PARAM_DIM)
        self.thinker_policy = WormPolicyNetwork(obs_dim, THINKER_HIDDEN_DIM,
                                                NUM_DISCRETE_ACTIONS, ACTION_PARAM_DIM)

    def get_action(self, obs: torch.Tensor, worm_type: torch.Tensor,
                   deterministic: bool = False):
        batch_size = obs.shape[0]
        actions = torch.zeros(batch_size, 1 + ACTION_PARAM_DIM, device=obs.device)
        log_probs = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, 1, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)

        worker_mask = (worm_type == 0)
        if worker_mask.any():
            a, lp, v, e = self.worker_policy.get_action(obs[worker_mask], deterministic)
            actions[worker_mask] = a
            log_probs[worker_mask] = lp
            values[worker_mask] = v
            entropies[worker_mask] = e

        thinker_mask = (worm_type == 1)
        if thinker_mask.any():
            a, lp, v, e = self.thinker_policy.get_action(obs[thinker_mask], deterministic)
            actions[thinker_mask] = a
            log_probs[thinker_mask] = lp
            values[thinker_mask] = v
            entropies[thinker_mask] = e

        return actions, log_probs, values, entropies

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         worm_type: torch.Tensor):
        batch_size = obs.shape[0]
        log_probs = torch.zeros(batch_size, device=obs.device)
        values = torch.zeros(batch_size, 1, device=obs.device)
        entropies = torch.zeros(batch_size, device=obs.device)

        worker_mask = (worm_type == 0)
        if worker_mask.any():
            lp, v, e = self.worker_policy.evaluate_actions(obs[worker_mask], actions[worker_mask])
            log_probs[worker_mask] = lp
            values[worker_mask] = v
            entropies[worker_mask] = e

        thinker_mask = (worm_type == 1)
        if thinker_mask.any():
            lp, v, e = self.thinker_policy.evaluate_actions(obs[thinker_mask], actions[thinker_mask])
            log_probs[thinker_mask] = lp
            values[thinker_mask] = v
            entropies[thinker_mask] = e

        return log_probs, values, entropies

    def get_all_parameters(self):
        return itertools.chain(self.worker_policy.parameters(),
                               self.thinker_policy.parameters())


class FloodPolicy(nn.Module):
    """
    Policy network for Flood agents.

    Simpler than Lekgolo: 32 hidden units.
    5 discrete actions: move, attack, infect, split, signal
    4 continuous params.

    Flood are deliberately less intelligent individually but
    compensate with numbers and replication advantage.
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.policy_net = WormPolicyNetwork(
            obs_dim, FLOOD_HIDDEN_DIM,
            FLOOD_NUM_DISCRETE_ACTIONS, FLOOD_ACTION_PARAM_DIM
        )

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        return self.policy_net.get_action(obs, deterministic)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                         worm_type: torch.Tensor | None = None):
        """Evaluate actions. worm_type is ignored (all Flood are same type)."""
        return self.policy_net.evaluate_actions(obs, actions)

    def get_all_parameters(self) -> list:
        return list(self.policy_net.parameters())
