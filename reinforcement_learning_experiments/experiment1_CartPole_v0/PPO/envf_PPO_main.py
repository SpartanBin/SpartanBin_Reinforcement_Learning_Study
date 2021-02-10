import time
import random

import numpy as np
import torch
import gym

from reinforcement_learning_experiments.experiment1_CartPole_v0.PPO import PPO_algorithm

# if gpu is to be used
device = 'cpu'

seed = 4000
random.seed(seed)
# Seed numpy RNG
np.random.seed(seed)
# seed the RNG for all devices (both CPU and CUDA)
torch.manual_seed(seed)

if device == 'cuda':
    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

env = gym.make('CartPole-v0')
env.seed(seed)
env = env.unwrapped

model = PPO_algorithm.PPO(env, device=device, seed=seed)
model.learn(total_timesteps=40000)

print('Have already been running {} episodes, now beginning valid'.format(model.episodes))

obs = env.reset()
n = 1
episode = 0
sum_episode_length = 0
while True:
    action = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        time.sleep(1)
        obs = env.reset()
        env.render()
        sum_episode_length += n
        episode += 1
        print(sum_episode_length / episode)
        n = 1
    else:
        n += 1