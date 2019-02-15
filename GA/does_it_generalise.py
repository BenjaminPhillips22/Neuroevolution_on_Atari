import os
import json
import time
import numpy as np
import pandas as pd
import random
import pickle
import glob

import atari_model


def random_seed_generator(seed=2):
    while True:
        random.seed(seed)
        new_rand = random.randint(0, 2**31-1)
        yield new_rand
        seed = new_rand


def id_generator(start=0):
    while True:
        start += 1
        yield start


def test_seed(seed_dict, config, m_id, env_seed, num_trails=12, monitor=True):
    """
    test if the model generalised or is only good on the specific
    env it stumbled upon. Creates a csv for each seed_dict and
    can make an mp4 for the best when monitor=True
    """
    our_rewards = []
    our_frames = []
    our_env_seeds = []

    seed = env_seed
    rsg = random_seed_generator(seed)

    model = atari_model.AtariModel(seed_dict, config)

    for n in range(num_trails):
        reward, frames, env_seed = model.evaluate_model(monitor=monitor,
                                                        set_env_seed=seed)
        seed = next(rsg)
        our_rewards.append(reward)
        our_frames.append(frames)
        our_env_seeds.append(env_seed)
        print('ID: ' + str(m_id) + ' run: ' + str(n), flush=True)

    csv_path = config['output_fname'] + "/test_seed_" + str(m_id) + '.csv'

    pd.DataFrame(
        {
            'tournament_env_seed': our_env_seeds,
            'reward': our_rewards,
            'frames': our_frames
        }
        ).to_csv(csv_path, index=False)


def main():

    FOLDER_NAME = 'v/frostbite-experiment-1550192228.0179362'

    # open json
    f_name = 'ga_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)

    # add time stamp to output_fname
    config['output_fname'] = FOLDER_NAME

    # start timing
    # start = time.time()

    # set run seed and add generators to config
    # config['get_id'] = id_generator()
    config['get_seed'] = random_seed_generator(config['run_seed'])

    # move into directory
    os.chdir(FOLDER_NAME)

    # Load data (deserialize)
    with open('tournament_winners.pickle', 'rb') as handle:
        tournament_winning_seed_dicts = pickle.load(handle)

    # get results from run
    files = glob.glob('*results.csv')
    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list).sort_values('reward', ascending=False)

    # check generalisability for top __
    checked_ids = []
    for i in range(5):
        m_id = df.iloc[i]['id']
        if m_id in checked_ids:
            pass
        elif m_id in tournament_winning_seed_dicts.keys():
            checked_ids.append(m_id)
            test_seed(seed_dict=tournament_winning_seed_dicts[m_id],
                      config=config,
                      m_id=m_id,
                      env_seed=df.iloc[i]['tournament_env_seed'])


if __name__ == "__main__":
    main()
