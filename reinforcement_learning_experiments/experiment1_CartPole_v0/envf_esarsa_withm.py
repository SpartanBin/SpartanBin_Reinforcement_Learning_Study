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

Transition = namedtuple('Transition', ('state', 'action', 'eps_threshold', 'next_state', 'reward'))


class ReplayMemory(object):
    '''
    注意memory储存的不是episode，而是元组('state', 'action', 'next_state', 'reward')
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # 相当于是个队列，在填满了样本后，再填东西会扔掉最前面填的
        if len(self.memory) < self.capacity:  # 填占位符，直到等于capacity
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # index到capacity后会归0

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = num_episodes / 2

# Get number of actions from gym action space
n_actions = env.action_space.n


nn_net = multi_layers_nn().to(device)

optimizer = optim.RMSprop(nn_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    next_eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done + 1) / EPS_DECAY)
    next_eps_threshold = torch.tensor([next_eps_threshold], device=device)
    steps_done += 1
    if sample > eps_threshold:
        nn_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_tensor = nn_net(state).max(1)[1].view(1, 1)
        nn_net.train()
        return (action_tensor, next_eps_threshold)
    else:
        return (torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), next_eps_threshold)


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
            '/exercise1_CartPole_v0/gym_env_observation_expected_sarsa/Duration_balance_pole_episode{}.jpg'.format(
                save_name))
        return None

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(
        map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    eps_threshold_batch = torch.cat(batch.eps_threshold)
    reward_batch = torch.cat(batch.reward)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # 初始化为0，最终状态的Q就为0
    nn_net.eval()
    with torch.no_grad():
        next_state_Q_value = nn_net(non_final_next_states)
        next_state_values[non_final_mask] = next_state_Q_value.max(1)[0] * (
            1 - eps_threshold_batch[non_final_mask] * (n_actions - 1) / n_actions) + \
            next_state_Q_value.min(1)[0] * eps_threshold_batch[non_final_mask] / n_actions

    # Compute the expected Sarsa values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    nn_net.train()
    state_action_values = nn_net(state_batch).gather(1, action_batch)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in nn_net.parameters():
        param.grad.data.clamp_(-1, 1)  # 这是在干嘛？是梯度裁剪
    optimizer.step()


def observation_to_tensor(observation):
    return torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)


for i_episode in range(num_episodes):
    # Initialize the environment and state
    observation = env.reset()
    state = observation_to_tensor(observation)
    for t in count():
        # Select and perform an action
        get_screen()
        action, eps_threshold = select_action(state)
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
        memory.push(state, action, eps_threshold, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 100 == 0:
                plot_durations(save_name='{}'.format(i_episode + 1))
            else:
                plot_durations()
            break

print('Complete')