'''
This script implements the classical algorithm named REINFORCE which belongs to policy gradient
'''

import sys
import os
import gym
import math
import random
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        '''

        :return: torch.tensor, this model's output must be probability for selecting action
        '''

        # x = F.elu(self.bn1(self.nn1(x)))
        # x = F.elu(self.bn2(self.nn2(x)))
        x = F.elu(self.nn1(x))
        x = F.elu(self.nn2(x))
        output = self.nn_output(x)
        return F.softmax(input=output, dim=1)


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
lr = 0.01

# Get number of actions from gym action space
n_actions = env.action_space.n


nn_net = multi_layers_nn().to(device)


def select_action(state, need_cal_grad):
    '''
    If use probability of chosen action to calculate gradient
    need output probability of chosen action

    :param state: torch.tensor
    :param need_cal_grad: bool, if need chosen action's gradient, set it to be True
    :return:
    '''
    if type(state) == list:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    # action_tensor need contain gradient
    action_tensor = nn_net(state)[0]
    action = torch.multinomial(action_tensor, num_samples=1).item()
    grad_list = []
    if need_cal_grad == True:
        # let parameters' grad be zero
        for param in nn_net.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        # calculate chosen action log's grad
        target_func = torch.log(action_tensor[action])
        target_func.backward()
        nn_net.to('cpu')
        for param in nn_net.parameters():
            param.grad.clamp_(-1, 1)
            grad_list.append(copy.deepcopy(param.grad))
        nn_net.to(device)

    return action, grad_list


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
            '/exercise1_CartPole_v0/experiment11/Duration_balance_pole_episode{}.jpg'.format(
                save_name))
        return None

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model(one_episode_reward, one_episode_grad):
    '''

    :param one_episode_reward: list
    :param one_episode_grad: list, length = len(one_episode_reward), item in one_episode_grad is
    a list which contains model parameters' grad. parameters' grad, torch.tensor, is in cpu
    :return:
    '''

    # 计算episode中每个state的回报G
    all_G = []
    all_GAMMA = []
    G = 0
    for i, reverse_i in enumerate(range(len(one_episode_reward) - 1, -1, -1)):
        G = GAMMA * G + one_episode_reward[reverse_i]
        all_G.append(G)
        all_GAMMA.append(GAMMA ** i)
    all_G.reverse()
    all_G = torch.tensor(all_G, dtype=torch.float32)
    all_GAMMA = torch.tensor(all_GAMMA, dtype=torch.float32)

    param_grad_mean = []
    for _ in one_episode_grad[0]:
        param_grad_mean.append([])
    for one_sample_grad_list in one_episode_grad:
        for i, one_sample_grad in enumerate(one_sample_grad_list):
            param_grad_mean[i].append(one_sample_grad.unsqueeze(0))
    for i, param_grad in enumerate(param_grad_mean):
        param_grad = torch.cat(param_grad, dim=0)
        all_G = all_G.view([all_G.shape[0]] + [1] * (len(param_grad.shape) - 1))
        all_GAMMA = all_GAMMA.view([all_GAMMA.shape[0]] + [1] * (len(param_grad.shape) - 1))
        param_grad = param_grad * all_GAMMA * all_G
        param_grad = param_grad.mean(dim=0)
        param_grad_mean[i] = param_grad

    # update model's params
    with torch.no_grad():  # When you want to update parameters, must set torch.no_grad()
        for i, param in enumerate(nn_net.parameters()):
            grad_mean = param_grad_mean[i].to(device)
            param += lr * grad_mean


for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset().tolist()
    one_episode_reward = []
    one_episode_grad = []
    for t in count():
        # Select and perform an action
        get_screen()
        action, grad_list = select_action(state, need_cal_grad=True)
        state, reward, done, _ = env.step(action)
        state = state.tolist()
        one_episode_reward.append(reward)
        one_episode_grad.append(grad_list)
        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 100 == 0:
                plot_durations(save_name='{}'.format(i_episode + 1))
            else:
                plot_durations()
            break
    optimize_model(
        one_episode_reward=one_episode_reward,
        one_episode_grad=one_episode_grad
    )

print('Complete')