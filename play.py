import sys
import cv2
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

    def reset(env):
        return convert_state(env.reset())

    def convert_state(state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

    env = gym.make('Frostbite-v4')
    
    # observation = env.reset()
    cur_states = [reset(env)] * 4
    old_state = cur_states.copy()

    for t in range(20):

        action = env.action_space.sample()  # select action randomly
        observation, reward, done, info = env.step(action)

        # update current state
        cur_states.pop(0)
        new_frame = convert_state(observation)
        cur_states.append(new_frame)

        assert old_state[1:4] == cur_states[0:3]
        old_state = cur_states.copy()

    env.env.close()
    print('all good :)')


def frostbite_simple_model(monitor=False, render=True):
    """
    Try getting a simple model to run
    """

    def reset(env):
        return convert_state(env.reset())

    def convert_state(state):
        return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

    env = gym.make('Frostbite-v4')
    if monitor:
        env = wrappers.Monitor(env, 'v/ffrostbite-experiment-' + str(time.time()))

    # simple model....

    
    # observation = env.reset()
    cur_states = [reset(env)] * 4
    total_reward = 0
    total_frames = 0

    for t in range(200):

        total_frames += 4

        #  model output
        # values = model(Variable(torch.Tensor([cur_states])))[0] 
        # action = np.argmax(values.data.numpy()[:env.action_space.n])
        # print(action)
        
        if render:
            env.render()
            time.sleep(0.05)

        # action = env.action_space.sample()  # select action randomly
        observation, reward, done, info = env.step(action)

        # update current state
        cur_states.pop(0)
        new_frame = convert_state(observation)
        cur_states.append(new_frame)

    env.env.close()


if __name__ == "__main__":
    """
    Run whatever function is called from the command line
    """
    locals()[sys.argv[1]]()

    # cartpole_no_model()
    # frostbite_no_model()
