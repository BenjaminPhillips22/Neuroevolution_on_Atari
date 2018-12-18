import sys
import time
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    """
    Run whatever function is called from the command line
    """
    locals()[sys.argv[1]]()

    # cartpole_no_model()
    # frostbite_no_model()
