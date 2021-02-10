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
        # self.bn1 = nn.BatchNorm1d(3)
        self.nn2 = nn.Linear(3, 2)
        # self.bn2 = nn.BatchNorm1d(2)
        self.nn_output = nn.Linear(2, 2)

    def forward(self, x):
        # x = F.elu(self.bn1(self.nn1(x)))
        # x = F.elu(self.bn2(self.nn2(x)))
        x = F.elu(self.nn1(x))
        x = F.elu(self.nn2(x))
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
TARGET_UPDATE = 10

# Get number of actions from gym action space
n_actions = env.action_space.n


policy_net = multi_layers_nn().to(device)
target_net = multi_layers_nn().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_tensor = policy_net(state).max(1)[1].view(1, 1)
        policy_net.train()
        return action_tensor
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


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
            '/exercise1_CartPole_v0/experiment7_part5/Duration_balance_pole_episode{}.jpg'.format(save_name))
        return None

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class one_batch_container(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)


def optimize_model(one_episode_sample):

    batch = Transition(*zip(*one_episode_sample))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(
        map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    if batch.next_state[0] is not None:
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        next_state_values = torch.zeros(len(one_episode_sample), device=device)  # 初始化为0，最终状态的Q就为0
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    else:
        next_state_values = 0

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Huber loss
    # To minimise this error, we will use the Huber loss.
    # The Huber loss acts like the mean squared error when the error is small,
    # but like the mean absolute error when the error is large -
    # this makes it more robust to outliers when the estimates of Q are very noisy.
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # 这是在干嘛？是梯度裁剪
    optimizer.step()


def observation_to_tensor(observation):
    return torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)


# 错误的实现
# for i_episode in range(num_episodes):
#     # Initialize the environment and state
#     one_batch = one_batch_container()
#     observation = env.reset()
#     state = observation_to_tensor(observation)
#     for t in count():
#         # Select and perform an action
#         get_screen()
#         action = select_action(state)
#         observation, reward, done, _ = env.step(action.item())
#         # if done:
#         #     reward = -100
#         reward = torch.tensor([reward], device=device)
#
#         # Observe new state
#         if not done:
#             next_state = observation_to_tensor(observation)
#         else:
#             next_state = None
#
#         # Store the transition in memory
#         one_batch.push(state, action, next_state, reward)
#
#         # Move to the next state
#         state = next_state
#
#         # Perform one step of the optimization (on the target network)
#         if done:
#             episode_durations.append(t + 1)
#             if (i_episode + 1) % 100 == 0:
#                 plot_durations(save_name='{}'.format(i_episode + 1))
#             else:
#                 plot_durations()
#             break
#     optimize_model(one_batch.memory)
#     # Update the target network, copying all weights and biases in DQN
#     if (i_episode + 1) % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

for i_episode in range(num_episodes):
    # Initialize the environment and state
    observation = env.reset()
    state = observation_to_tensor(observation)
    for t in count():
        # Select and perform an action
        get_screen()
        one_batch = one_batch_container()
        action = select_action(state)
        observation, reward, done, _ = env.step(action.item())
        # if done:
        #     reward = -100
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = observation_to_tensor(observation)
        else:
            next_state = None

        # Store the transition in memory
        one_batch.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        optimize_model(one_batch.memory)
        # Perform one step of the optimization (on the target network)
        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 100 == 0:
                plot_durations(save_name='{}'.format(i_episode + 1))
            else:
                plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if (i_episode + 1) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')