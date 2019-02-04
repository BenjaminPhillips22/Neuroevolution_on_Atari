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
        
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)

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
        # self.model_type = config['model']
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

    f_name = 'rs_tests/colab_config.json'

    with open(f_name) as f:
        config = json.load(f)
    
    config['output_fname'] = config['output_fname'] + '-s2v-' + str(time.time())

    # get best seed
    print('Checking Rep', flush=True)
    best_seed = 270687994  # 2  #2056784773  #270687994 # our_seeds[np.argmax(our_rewards)]
    print('seed: ', best_seed)
    for _ in range(3):
        m = RSModel(seed=best_seed, config=config)
        r, f = m.evaluate_model(monitor=False)
        print('reward: ', str(r), ' frames: ', str(f))

    for i in range(3):
        config['output_fname'] = config['output_fname'] + '_' + str(i)
        m = RSModel(seed=best_seed, config=config)
        r, f = m.evaluate_model(monitor=True)
        print('reward: ', str(r), ' frames: ', str(f))


    elapsed = (time.time() - start)
    print("Time: " + str(round(elapsed)))

    # save our results
    # This will only work if the dir has been created above.
    # csv_path = config['output_fname'] + '/results.csv'
    # pd.DataFrame(
    #     {
    #         'seed': [best_seed],
    #         'reward': [r],
    #         'time': [elapsed],
    #         'frames': [f]
    #     }
    #     ).to_csv(csv_path)


if __name__ == "__main__":
    main()