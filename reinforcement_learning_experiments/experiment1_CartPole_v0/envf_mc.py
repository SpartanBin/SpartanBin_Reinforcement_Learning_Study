import sys
import os
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class multi_layers_nn(nn.Module):

    def __init__(self):
        super(multi_layers_nn, self).__init__()
        self.nn1 = nn.Linear(4, 3)
        self.bn1 = nn.BatchNorm1d(3)
        self.nn2 = nn.Linear(3, 2)
        self.bn2 = nn.BatchNorm1d(2)
        self.nn_output = nn.Linear(2, 2)

    def forward(self, x):
        x = F.elu(self.bn1(self.nn1(x)))
        x = F.elu(self.bn2(self.nn2(x)))
        output = self.nn_output(x)
        return output


def get_screen():
    return env.render(mode='rgb_array')  # HWC


env.reset()
# 画原始屏幕
# origin_screen = get_screen()
# plt.figure()
# plt.imshow(origin_screen, interpolation='none')
# plt.title('Example origin screen')
# plt.savefig(project_path + '/exercise1_CartPole_v0/DQN_change_state_features/Example_origin_screen.jpg')
# plt.close()


num_episodes = 20000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = num_episodes / 2

# Get number of actions from gym action space
n_actions = env.action_space.n


nn_net = multi_layers_nn().to(device)

optimizer = optim.RMSprop(nn_net.parameters())


steps_done = 0


def select_action(state):
    if type(state) == list:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        nn_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_tensor = nn_net(state).max(1)[1].view(1, 1).to('cpu').tolist()
        nn_net.train()
        return action_tensor[0][0]
    else:
        return random.randrange(n_actions)


episode_durations = []


def plot_durations(save_name=None):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if save_name is not None:
        plt.savefig(
            project_path +
            '/exercise1_CartPole_v0/gym_env_observation_mc/Duration_balance_pole_episode{}.jpg'.format(
                save_name))
        return None

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model(one_episode_sample):
    '''

    :param one_episode_sample: torch.tensor, shape = (episode_length, value_dim),
    the dim of value = (state1, state2, state3, state4, action, reward), 不能存入GPU
    注意one_episode中不能包含最终状态，因为最终状态有state_features但是没有action，因为已经失败了
    :return:
    '''

    state_features = one_episode_sample[:, : 4].to(device)
    action = one_episode_sample[:, 4].to(torch.int64).unsqueeze(1).to(device)
    reward = one_episode_sample[:, 5]

    # 计算episode中每个state的回报G
    all_G = []
    G = 0
    for i in range(len(one_episode_sample) - 1, -1, -1):
        G = GAMMA * G + reward[i]
        all_G.append(G)
    all_G.reverse()
    all_G = torch.tensor(all_G, dtype=torch.float32).unsqueeze(1).to(device)

    # 依然注意动作价值函数是(a|s)的价值，是对在某特定状态下执行特定动作的评价
    state_action_values = nn_net(state_features).gather(1, action)

    loss = F.smooth_l1_loss(state_action_values, all_G)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in nn_net.parameters():
        param.grad.data.clamp_(-1, 1)  # 这是在干嘛？是梯度裁剪
    optimizer.step()


for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset().tolist()
    one_episode_sample = []
    for t in count():
        # Select and perform an action
        get_screen()
        action = select_action(state)
        observation_action_reward = state + [action]
        state, reward, done, _ = env.step(action)
        state = state.tolist()
        # if done:
        #     reward = -100
        observation_action_reward.append(reward)
        one_episode_sample.append(observation_action_reward)
        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 100 == 0:
                plot_durations(save_name='{}'.format(i_episode + 1))
            else:
                plot_durations()
            break
    one_episode_sample = torch.tensor(one_episode_sample, dtype=torch.float32)
    optimize_model(one_episode_sample)

print('Complete')