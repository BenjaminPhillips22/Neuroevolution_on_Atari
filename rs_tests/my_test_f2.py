import os
import time
import json
import pandas as pd

from multiprocessing import Queue, Process
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import cv2

import gym
from gym import wrappers


def get_rewards(queue, seed, config):
    # torch.set_num_threads(1)
    m = RSModel(seed=seed, config=config)
    reward, frames = m.evaluate_model()
    queue.put([seed, reward, frames])


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
        self.model = BigModel(seed).cuda()

    def convert_state(self, state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

    def reset(self, env):
        return self.convert_state(env.reset())

    def evaluate_model(self, monitor=False):

        env = gym.make(self.env_name)

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
            # not sure about whether to use cuda() on cur_states or
            # torch.Tensor([cur_states]).
            # https://medium.com/@layog/a-comprehensive-overview-of-pytorch-7f70b061963f 
            values = self.model(Variable(torch.Tensor([cur_states]).cuda()))[0]
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

    assert torch.cuda.is_available(), "GPU is not available!!!!!!!!"
        
    # start timing
    start = time.time()

    f_name = 'rs_tests/test_a1.json'

    our_seeds = []
    our_rewards = []
    # our_time = []
    our_frames = []
    total_frames = 0

    with open(f_name) as f:
        config = json.load(f)
    
    config['output_fname'] = config['output_fname'] + 'e1-' + str(time.time())

    q = Queue()
    processes = []
    for s in range(60):
        p = Process(target=get_rewards, args=(q, s, config))
        processes.append(p)
        p.start()
    for p in processes:
        seed, reward, frames = q.get()  # will block
        our_seeds.append(seed)
        our_rewards.append(reward)
        our_frames.append(frames)
    for p in processes:
        p.join()

    elapsed = (time.time() - start)

    if not os.path.isdir(config['output_fname']):
        os.mkdir(config['output_fname'])

    csv_path = config['output_fname'] + '/results.csv'
    pd.DataFrame(
        {
            'seed': our_seeds,
            'reward': our_rewards,
            'frames': our_frames,
            'elapsed': elapsed
        }
        ).to_csv(csv_path)

    print("e1 Time: " + str(round(elapsed)))


if __name__ == "__main__":
    main()
