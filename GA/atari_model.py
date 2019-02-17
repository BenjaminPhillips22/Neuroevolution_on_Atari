# In the GA folder
# 

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import gym
from gym import wrappers
import random

import base_model


class AtariModel():
    """
    Takes config which has atari game parameters.
    Also takes seed_dict which is passed to BigModel.
    """
    def __init__(self, seed_dict, config):
        self.get_seed = config['get_seed']
        self.env_name = config['env']
        self.max_frames_per_episode = config['max_frames_per_episode']
        self.output_fname = config['output_fname']
        self.model = base_model.BigModel(seed_dict, config)

    def convert_state(self, state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

    def reset(self, env):
        return self.convert_state(env.reset())

    def evaluate_model(self, monitor=False, max_noop=30, set_env_seed=False, output_fn='AUTO'):
        """
        outputs reward, frames and env_seed. Can create an mp3 with monitor=True.
        set_env_seed is False if not setting a seed, or the number of the seed you
        want to set (eg set_env_seed=101) if you do want to set the env_seed.
        output_fn allows user to select output folder for mp4 otherwise the folder set
        in the config is used.
        """
        env = gym.make(self.env_name)

        if not set_env_seed:
            env_seed = next(self.get_seed)
        else:
            env_seed = set_env_seed

        env.seed(env_seed)

        cur_states = [self.reset(env)] * 4
        total_reward = 0
        total_frames = 0
        old_lives = env.env.ale.lives()

        if monitor:
            if output_fn == 'AUTO':
                output_fn = self.output_fname
            else:
                pass
            env = wrappers.Monitor(env, output_fn)

        env.reset()

        # implement random no-operations
        random.seed(env_seed)
        noops = random.randint(0, max_noop)

        for _ in range(noops):
            observation, reward, done, _ = env.step(0)
            total_reward += reward
            if done:
                break
            cur_states.pop(0)
            new_frame = self.convert_state(observation)
            cur_states.append(new_frame)

        for t in range(self.max_frames_per_episode):

            total_frames += 4

            #  model output
            values = self.model(Variable(torch.Tensor([cur_states])))[0]
            action = np.argmax(values.data.numpy()[:env.action_space.n])
            observation, reward, done, _ = env.step(action)

            # update current state
            total_reward += reward

            if monitor:
                new_lives = env.env.env.ale.lives()
                if old_lives < new_lives:
                    break
            else:
                new_lives = env.env.ale.lives()
                if old_lives < new_lives:
                    break
            old_lives = new_lives

            # break if it's been one life and 0 reward
            # need to be careful with this, it won't generalise to other games
            if old_lives == 3:
                if total_reward == 0:
                    break

            if done:
                break
            cur_states.pop(0)
            new_frame = self.convert_state(observation)
            cur_states.append(new_frame)

        env.env.close()
        return total_reward, total_frames, env_seed
