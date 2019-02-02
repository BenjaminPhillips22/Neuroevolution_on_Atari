import time
import json
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt

import gym
from gym import wrappers


class BigModel(nn.Module):
    """
    A big model....
    """
    def __init__(self, seed):
        super().__init__()

        self.seed = seed
        torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(4, 32, (8, 8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.dense = nn.Linear(4*4*64, 512)
        self.out = nn.Linear(512, 18)

        self.add_tensors = {}
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, -1)
        x = F.relu(self.dense(x))
        return self.out(x)


class RSModel():

    def __init__(self, seed, config):
        self.env_name = config['env']
        self.max_frames_per_episode = config['max_frames_per_episode']
        self.output_fname = config['output_fname']
        self.model_type = config['model']
        # below is equivalent to models.SmallModel(seed)
        # self.model = getattr(models, self.model_type)(seed)
        self.model = BigModel(seed)

    def convert_state(self, state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

    def reset(self, env):
        return self.convert_state(env.reset())

    def evaluate_model(self, monitor=False):

        env = gym.make(self.env_name)
        env.seed(0)

        cur_states = [self.reset(env)] * 4
        total_reward = 0
        total_frames = 0
        old_lives = env.env.ale.lives()

        if monitor:
            env = wrappers.Monitor(env, self.output_fname)

        env.reset()

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

    # start timing
    start = time.time()

    f_name = 'rs_tests/test_a1.json'

    our_seeds = []
    our_rewards = []
    our_time = []
    our_frames = []
    total_frames = 0

    with open(f_name) as f:
        config = json.load(f)
    
    config['output_fname'] = config['output_fname'] + '-c3-' + str(time.time())

    # start the 'random' search!
    for s in range(6):
        
        m = RSModel(seed=s, config=config)
        reward, frames = m.evaluate_model()
        
        elapsed = (time.time() - start)

        our_time.append(elapsed)
        our_seeds.append(s)
        our_rewards.append(reward)
        our_frames.append(frames)
        total_frames += frames

        print("Time: " + str(round(elapsed)) +
              ", Frames: " + str(total_frames), flush=True)

    # get best seed
    print('recording best network', flush=True)
    best_seed = our_seeds[np.argmax(our_rewards)]
    m = RSModel(seed=best_seed, config=config)
    _, _ = m.evaluate_model(monitor=True)

    # save our results
    # This will only work if the dir has been created above.
    csv_path = config['output_fname'] + '/results.csv'
    pd.DataFrame(
        {
            'seed': our_seeds,
            'reward': our_rewards,
            'time': our_time,
            'frames': our_frames
        }
        ).to_csv(csv_path)

    elapsed = (time.time() - start)
    print("c3 Time: " + str(round(elapsed)))


if __name__ == "__main__":
    main()
