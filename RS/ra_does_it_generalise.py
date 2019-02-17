import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import torch
from torch.autograd import Variable

import gym
from gym import wrappers

import models


class RSModel():
    """
    Create a new RSModel here that has noops
    """

    def __init__(self, config):
        self.env_name = config['env']
        self.max_frames_per_episode = config['max_frames_per_episode']
        self.output_fname = config['output_fname']
        self.model = models.BigModel(config['run_seed'])

    def convert_state(self, state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

    def reset(self, env):
        return self.convert_state(env.reset())

    def evaluate_model(self, monitor=False, noops=0):

        env = gym.make(self.env_name)
        env.seed(noops)

        cur_states = [self.reset(env)] * 4
        total_reward = 0
        total_frames = 0
        old_lives = env.env.ale.lives()

        if monitor:
            env = wrappers.Monitor(env, self.output_fname)

        env.reset()

        # implement no-operations
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
        return total_reward, total_frames


def main():
    """
    Have to put seed in json
    """

    # start timing
    start = time.time()

    # open json
    f_name = 'rs_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)
    
    # add time to folder name
    config['output_fname'] = config['output_fname'] + 's2v-' + str(time.time())

    # creating a new folder
    os.makedirs(config['output_fname'])

    # saving config
    csv_path = config['output_fname'] + '/config.csv'
    pd.DataFrame.from_dict(config, orient='index').to_csv(csv_path)

    # get best seed
    print('starting processes', flush=True)

    # store info
    our_noops = []
    our_rewards = []
    our_frames = []

    # try different noops
    for n in range(0, 31, 2):

        m = RSModel(config=config)
        r, f = m.evaluate_model(noops=n)

        our_noops.append(n)
        our_rewards.append(r)
        our_frames.append(f)

        elapsed = (time.time() - start)
        print("Time: " + str(round(elapsed)),
              " noop: " + str(n), flush=True)

    # get best seed
    print('recording best network', flush=True)
    best_noop = our_noops[np.argmax(our_rewards)]
    m = RSModel(config=config)
    _, _ = m.evaluate_model(monitor=True, noops=best_noop)

    # save our results
    csv_path = config['output_fname'] + '/results.csv'
    pd.DataFrame(
        {
            'noops': our_noops,
            'reward': our_rewards,
            'frames': our_frames
        }
        ).to_csv(csv_path)

    # save boxplot of results
    ax = sns.boxplot(data=our_rewards, color="lightblue")
    ax = sns.swarmplot(data=our_rewards, color=".1")
    title = 'seed_' + str(config['run_seed'])
    ax.set_title(title)
    figure = ax.get_figure()
    plot_path = config['output_fname'] + '/' + title + '.png'
    figure.savefig(plot_path)


if __name__ == "__main__":
    main()
