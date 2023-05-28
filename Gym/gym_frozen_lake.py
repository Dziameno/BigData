import gymnasium as gym
import pygame
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
random_size = np.random.randint(4, 8)
random_floes = np.random.random()
random_map = generate_random_map(size=random_size, p=random_floes)

env = gym.make("FrozenLake-v1", render_mode="human", desc=random_map, is_slippery=True)
for run in range(10):
    env.reset()
    env.render()
    for step in range(100):
        action = np.random.randint(3)
        location, reward, isDead, info, terminated = env.step(action)
        print('Run:', run, 'Step:',step)
        print (location, reward, isDead, info, terminated)
        if reward == 1:
            print('Sucess!')
            #load new map
        if isDead == True:
            env.reset()











