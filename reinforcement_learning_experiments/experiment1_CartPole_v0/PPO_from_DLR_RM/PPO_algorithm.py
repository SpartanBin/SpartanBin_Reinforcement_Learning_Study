from typing import Optional, Union, Tuple

import numpy as np
import torch as th
import gym
from gym import spaces
import torch
from torch.nn import functional as F

from reinforcement_learning_experiments.experiment1_CartPole_v0.PPO_from_DLR_RM import buffers, policies


class PPO():

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cpu",
    ):

        self.num_timesteps = 0
        self.seed = seed
        self.lr_schedule = learning_rate
        self._current_progress_remaining = 1

        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.n_envs = 1

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.all_obs = []

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        self.rollout_buffer = buffers.RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = policies.ActorCriticPolicy(
            lr_schedule=self.lr_schedule,
        ).to(self.device)

    def collect_rollouts(
        self,
        env,
        rollout_buffer,
        n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param rollout_buffer: Buffer to fill with rollouts
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        dones = False
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:

            with th.no_grad():
                # Convert to pytorch tensor
                self.all_obs.append(self._last_obs.flatten())
                obs_tensor = th.as_tensor(self._last_obs).to(self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions[0])
            self.env_timestep += 1
            self.G += 1
            if dones:
                print(self.G)
                new_obs = self.env.reset()
                self.env_timestep = 0
                self.G = 1
            new_obs = new_obs.reshape(1, -1).astype(np.float32)

            self.num_timesteps += 1

            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        for param_group in self.policy.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule
        # Compute current clip range
        clip_range = self.clip_range
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # if that line is commented (as in SAC)

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

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, - clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                break

    def learn(
        self,
        total_timesteps: int):

        self.num_timesteps = 0
        self._last_obs = self.env.reset().reshape(1, -1).astype(np.float32)
        self._last_dones = False
        self.env_timestep = 0
        self.G = 1

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                env=self.env,
                rollout_buffer=self.rollout_buffer,
                n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(total_timesteps)

            self.train()

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        observation = torch.tensor(observation.reshape((-1, 4)).astype(np.float32)).to(self.device)
        with torch.no_grad():
            actions, values, log_prob = self.policy.forward(
                obs=observation,
                deterministic=deterministic
            )
        return actions.cpu().item()