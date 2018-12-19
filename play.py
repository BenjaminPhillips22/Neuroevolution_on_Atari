import sys
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

import utils


def cartpole_no_model():
    """
    Run and render cartpole.
    No model, just sample from action space for
    20 episodes, max frames per episode is 100.
    """
    env = gym.make('CartPole-v1')

    for i_episode in range(20):

        observation = env.reset()

        for t in range(100):

            env.render()
            time.sleep(0.05)

            action = env.action_space.sample()  # select actino randomly
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


def frostbite_no_model():
    """
    Run and render frostbite.
    No model, just sample from action space for
    1 episodes, max frames per episode is 1000.
    """
    env = gym.make('Frostbite-v0')

    for i_episode in range(1):

        observation = env.reset()

        for t in range(1000):

            env.render()
            time.sleep(0.05)

            action = env.action_space.sample()  # select actino randomly
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


def frostbite_no_model_monitor():
    """
    Run and record output of frostbite.
    No model, just sample from action space for
    1 episodes, max frames per episode is _.
    -v4 to make it deterministic.
    """
    env = gym.make('Frostbite-v4')
    env = wrappers.Monitor(env, 'v/ffrostbite-experiment-' + str(time.time()))

    for i_episode in range(1):

        observation = env.reset()

        for t in range(200):

            # uncomment below to render while recording
            # env.render()
            # time.sleep(0.05)

            action = env.action_space.sample()  # select action randomly
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.env.close()


def show_frame_change():
    """
    Show image before and after preprocessing
    """
    env = gym.make('Frostbite-v4')
    observation = env.reset()

    for i in range(75):
        if i == 0: env.step(1)
        env.step(0)

    state_before = env.render(mode='rgb_array')
    state_after = utils.convert_state(state_before)

    fig, axes = plt.subplots(1, 2)

    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)

    axes[0].imshow(state_before)
    axes[1].imshow(state_after)

    plt.show()

    env.env.close()


def test_cur_state():
    """
    Test if cur_state works they way we hope it does
    """
    env = gym.make('Frostbite-v4')
    
    # observation = env.reset()
    cur_states = [utils.reset(env)] * 4
    old_state = cur_states.copy()

    for t in range(20):

        action = env.action_space.sample()  # select action randomly
        observation, reward, done, info = env.step(action)

        # update current state
        cur_states.pop(0)
        new_frame = utils.convert_state(observation)
        cur_states.append(new_frame)

        assert old_state[1:4] == cur_states[0:3]
        old_state = cur_states.copy()

    env.env.close()
    print('all good :)')


def frostbite_simple_model(monitor=False, render=True):
    """
    Try getting a simple model to run
    """

    env = gym.make('Frostbite-v4')
    if monitor:
        env = wrappers.Monitor(env, 'v/ffrostbite-experiment-' + str(time.time()))

    class Model(nn.Module):
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

    model = Model(8)

    cur_states = [utils.reset(env)] * 4
    total_reward = 0
    total_frames = 0

    for t in range(200):

        total_frames += 4

        #  model output
        values = model(Variable(torch.Tensor([cur_states])))[0]
        action = np.argmax(values.data.numpy()[:env.action_space.n])
        print(action)
        observation, reward, done, info = env.step(action)

        if render:
            env.render()
            time.sleep(0.05)

        # update current state
        cur_states.pop(0)
        new_frame = utils.convert_state(observation)
        cur_states.append(new_frame)
        total_reward += reward

    env.env.close()
    print(total_reward)


def test_reproducible_random_seed():
    """
    Run a few models with identical random seeds and
    see if the reward and action sequence are the same.
    """
    env = gym.make('Frostbite-v4')

    random_seeds = list(range(1, 6))

    for i, j in zip(random_seeds, random_seeds):
        model_1 = utils.SmallModel(i)
        model_2 = utils.SmallModel(i)

        total_reward_m1, all_actions_m1 = utils.evaluate_model(env, model_1)
        total_reward_m2, all_actions_m2 = utils.evaluate_model(env, model_2)

        assert (total_reward_m1 == total_reward_m2) and (all_actions_m1 == all_actions_m2)

    env.env.close()
    print('all good :D')


def distribution_of_reward():
    """
    Plot the distribution of reward for many random seeds
    """
    random_seeds = list(range(60))
    reward_list = []

    for i in random_seeds:
        model = utils.SmallModel(i)
        total_reward, all_actions = utils.evaluate_model(model, render=False)
        reward_list.append(total_reward)
        # print(total_reward)

    fig, ax = plt.subplots()
    ax.hist(reward_list)
    plt.show()

    print('best seed is {}'.format(np.argmax(np.array(reward_list))))


def convert_video():
    """
    check what the model "sees"
    """
    import imageio
    fname = "C:/Users/ben/Documents/Projects/Neuroevolution_on_Atari/v/frostbite-experiment-1545199896.5494895/openaigym.video.0.5680.video000000.mp4"

    reader = imageio.get_reader(fname)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer('model_view.mp4', fps=fps)

    for im in reader:
        writer.append_data(utils.convert_state(im))
    writer.close()



if __name__ == "__main__":
    """
    Run whatever function is called from the command line
    """
    locals()[sys.argv[1]]()

    # cartpole_no_model()
    # frostbite_no_model()
