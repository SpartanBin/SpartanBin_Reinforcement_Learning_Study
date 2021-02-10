from typing import Optional, Union, Tuple

import numpy as np
import torch as th
import torch
from torch.nn import functional as F

from reinforcement_learning_experiments.experiment1_CartPole_v0.PPO import buffers, policies


class PPO():

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range=0.2,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cpu",
    ):

        self.num_timesteps = 0
        self.seed = seed
        self.learning_rate = learning_rate

        self.env = env

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.episodes = 1

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range

        self.rollout_buffer = buffers.RolloutBuffer(
            buffer_size=self.n_steps,
            obs_shape=(4, ),
            action_dim=1,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
        )
        self.policy = policies.ActorCriticPolicy(
            lr_schedule=self.learning_rate,
        ).to(self.device)

    def collect_rollouts(self) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        dones = False
        self.rollout_buffer.reset()

        while n_steps < self.n_steps:

            with th.no_grad():
                # Convert to pytorch tensor
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = self.env.step(actions[0])
            self.env_timestep += 1
            self.G += 1
            if dones:
                print(self.G)
                new_obs = self.env.reset()
                self.env_timestep = 0
                self.G = 1
                self.episodes += 1
            new_obs = new_obs.reshape(1, -1).astype(np.float32)

            self.num_timesteps += 1

            n_steps += 1

            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)
            self.rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        for param_group in self.policy.optimizer.param_groups:
            param_group["lr"] = self.learning_rate
        # Compute current clip range
        clip_range = self.clip_range

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.forward(rollout_data.observations, actions)
                values = th.flatten(values)
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)

                loss = policy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

    def learn(
        self,
        total_timesteps: int):

        self.num_timesteps = 0
        self._last_obs = self.env.reset().reshape(1, -1).astype(np.float32)
        self._last_dones = False
        self.env_timestep = 0
        self.G = 1

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts()

            if continue_training is False:
                break

            self.train()

    def predict(
        self,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        observation = torch.tensor(observation.reshape((-1, 4)).astype(np.float32)).to(self.device)
        with torch.no_grad():
            actions, values, log_prob = self.policy.forward(
                obs=observation,
            )
        return actions.cpu().item()