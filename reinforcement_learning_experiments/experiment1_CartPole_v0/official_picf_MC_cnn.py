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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DMC(nn.Module):

    def __init__(self, h, w, outputs):
        super(DMC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Resize((48, 180), interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    origin_screen = env.render(mode='rgb_array')
    screen = origin_screen.transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    # 它对宽度进行切片操作，得到一个以小车为中心的图片，会不会影响计算速度呢？？？？因为它的速度似乎是通过输入多张连续图片进入cnn，自动处理的
    # 它上面说的会输入历史图像序列进去，但是看网络结构，第一层卷积in_channel = 3，说明并没有输入历时图像序列，也就是说他并没有考虑速度的影响
    # 还是说这个任务里并没有小车速度的概念，我应该去查一下这个env的observation四个向量分量代表的含义
    # 但是还是可以确定这样裁剪图片相当于无视了position这个state
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    # 可以这样认为，ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device), origin_screen


env.reset()
# # 画提取的窗口
# plt.figure()
# slice_screen, origin_screen = get_screen()
# plt.imshow(slice_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
# plt.title('Example extracted screen')
# plt.savefig(project_path + '/exercise1_CartPole_v0/DQN_change_state_features/Example_extracted_screen.jpg')
# plt.close()
# # 画原始屏幕
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


init_screen, _ = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n


cnn_net = DMC(h=screen_height,
              w=screen_width,
              outputs=2).to(device)

optimizer = optim.RMSprop(cnn_net.parameters())


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
        cnn_net.eval()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_tensor = cnn_net(state).max(1)[1].view(1, 1).to('cpu').tolist()
        cnn_net.train()
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
            '/exercise1_CartPole_v0/experiment9/Duration_balance_pole_episode{}.jpg'.format(
                save_name))
        return None

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model(one_episode_sample):
    '''

    :param one_episode_sample: 是字典, 包含键 'state', 'action', 'reward'
    state: torch.tensor, shape = (episode_length, C, H, W)
    action: torch.tensor, shape = (episode_length)
    reward: torch.tensor, shape = (episode_length)
    注意one_episode中不能包含最终状态，因为最终状态有state_features但是没有action，因为已经失败了
    :return:
    '''

    state_features = one_episode_sample['state'].to(device)
    action = one_episode_sample['action'].to(torch.int64).unsqueeze(1).to(device)
    reward = one_episode_sample['reward']

    # 计算episode中每个state的回报G
    all_G = []
    G = 0
    for i in range(len(action) - 1, -1, -1):
        G = GAMMA * G + reward[i]
        all_G.append(G)
    all_G.reverse()
    all_G = torch.tensor(all_G, dtype=torch.float32).unsqueeze(1).to(device)

    # 依然注意动作价值函数是(a|s)的价值，是对在某特定状态下执行特定动作的评价
    state_action_values = cnn_net(state_features).gather(1, action)

    loss = F.smooth_l1_loss(state_action_values, all_G)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in cnn_net.parameters():
        param.grad.data.clamp_(-1, 1)  # 这是在干嘛？是梯度裁剪
    optimizer.step()


for i_episode in range(num_episodes):

    # Initialize the environment and state
    env.reset()
    last_screen, _ = get_screen()
    current_screen, origin_screen = get_screen()
    state = current_screen - last_screen
    one_episode_sample = {
        'state': [],
        'action': [],
        'reward': []
    }

    for t in count():

        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action)
        one_episode_sample['state'].append(state)
        one_episode_sample['action'].append(action)
        one_episode_sample['reward'].append(reward)

        # Observe new state
        last_screen = current_screen
        current_screen, origin_screen = get_screen()
        state = current_screen - last_screen

        if done:
            episode_durations.append(t + 1)
            if (i_episode + 1) % 5000 == 0:
                plot_durations(save_name='{}'.format(i_episode + 1))
            else:
                plot_durations()
            break

    one_episode_sample['state'] = torch.cat(one_episode_sample['state'], dim=0)
    one_episode_sample['action'] = torch.tensor(one_episode_sample['action'], dtype=torch.float32)
    one_episode_sample['reward'] = torch.tensor(one_episode_sample['reward'], dtype=torch.float32)
    optimize_model(one_episode_sample)

print('Complete')