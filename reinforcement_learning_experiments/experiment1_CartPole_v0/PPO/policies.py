"""Policies: abstract base class and concrete implementations."""

from functools import partial
from typing import Tuple, Union

import numpy as np
import torch as th
from torch import nn


class MlpExtractor(nn.Module):

    def __init__(self):
        super(MlpExtractor, self).__init__()

        policy_net = []
        value_net = []

        feature_dims = (4, 64, 64)
        for i in range(len(feature_dims)):
            if i > 0:
                policy_net.append(nn.Linear(feature_dims[i - 1], feature_dims[i]))
                policy_net.append(nn.Tanh())
                value_net.append(nn.Linear(feature_dims[i - 1], feature_dims[i]))
                value_net.append(nn.Tanh())

        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class ActorCriticPolicy(nn.Module):

    def __init__(
        self,
        lr_schedule,
        ortho_init: bool = True
    ):
        super(ActorCriticPolicy, self).__init__()

        self.lr_schedule = lr_schedule
        self.ortho_init = ortho_init

        self.mlp_extractor = MlpExtractor()
        self.value_net = nn.Linear(64, 1)
        self.action_net = nn.Linear(64, 2)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.lr_schedule, eps=1e-5)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(
            self,
            obs: th.Tensor,
            actions: Union[th.Tensor, None] = None):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.mlp_extractor(obs)
        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        ac_prob = th.softmax(mean_actions, dim=1)
        if actions is None:
            actions = th.multinomial(ac_prob[0], num_samples=1)
            log_prob = th.log(ac_prob)[:, actions[0]]
            return actions, values, log_prob
        else:
            log_prob = th.log(ac_prob).gather(1, actions.unsqueeze(1)).flatten()
            entropy = 0
            return values, log_prob, entropy