import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import gym
from gym import wrappers

import matplotlib.pyplot as plt


def reset(env):
    return convert_state(env.reset())


def convert_state(state):
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0


class SmallModel(nn.Module):
    """
    A simple model....
    """
    def __init__(self, rng_state):
        super().__init__()

        # TODO: padding?
        self.rng_state = rng_state
        torch.manual_seed(rng_state)

        self.conv1 = nn.Conv2d(4, 16, (8, 8), 4)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), 2)
        self.dense = nn.Linear(1152, 64)
        self.out = nn.Linear(64, 18)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))
        return self.out(x)


def evaluate_model(model, episode_length=200, render=False):

    env = gym.make('Frostbite-v4')

    cur_states = [reset(env)] * 4
    total_reward = 0
    all_actions = []
    total_frames = 0

    for t in range(episode_length):

        total_frames += 4

        if render:
            env.render()
            time.sleep(0.05)

        #  model output
        values = model(Variable(torch.Tensor([cur_states])))[0]
        action = np.argmax(values.data.numpy()[:env.action_space.n])
        all_actions.append(action)
        observation, reward, done, info = env.step(action)

        # update current state
        cur_states.pop(0)
        new_frame = convert_state(observation)
        cur_states.append(new_frame)
        total_reward += reward

    env.env.close()
    return total_reward, all_actions
