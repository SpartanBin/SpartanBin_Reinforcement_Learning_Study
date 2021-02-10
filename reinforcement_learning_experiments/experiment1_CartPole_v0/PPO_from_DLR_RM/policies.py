"""Policies: abstract base class and concrete implementations."""

from functools import partial
from typing import Tuple, Union

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical


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


class CategoricalDistribution():
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


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
        # Action distribution
        self.action_distribution = CategoricalDistribution(2)
        self.action_net = self.action_distribution.proba_distribution_net(latent_dim=64)

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

    def output_action_distribution(self, latent_pi: th.Tensor):
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        return self.action_distribution.proba_distribution(action_logits=mean_actions)

    def forward(
            self,
            obs: th.Tensor,
            actions: Union[th.Tensor, None] = None,
            deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        latent_pi, latent_vf = self.mlp_extractor(obs)
        values = self.value_net(latent_vf)
        distribution = self.output_action_distribution(latent_pi)
        if actions is None:
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            return actions, values, log_prob
        else:
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return values, log_prob, entropy