import numpy as np
import gym

env = gym.make('CartPole-v0')


# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())  # take a random action
# env.close()


# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()


# env.reset()
# for _ in range(1000):
#     # env.step(env.action_space.sample())
#     print(env.action_space.sample())
#     print(env.observation_space.sample())


# env = gym.make("CartPole-v1")
observation = env.reset()
print(observation)
#
# sc = env.render(mode='rgb_array').transpose((2, 0, 1))
# print(np.unique(sc))

# for _ in range(10):
#   # env.render()
#   action = env.action_space.sample() # your agent here (this takes random actions)
#   print(action)
#   # observation, reward, done, info = env.step(action)
#   # print(observation)
#   # break
#   # if info:
#   #   print(info)
#   #   break
#
#   # if done:
#   #   observation = env.reset()
# env.close()