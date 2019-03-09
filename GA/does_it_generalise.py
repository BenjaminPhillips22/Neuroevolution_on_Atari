"""
Run the seeds through their paces.
Will have to update code if using winners from ga_rerun.py
"""

import os
import json
import time
import seaborn as sns
import numpy as np
import pandas as pd
import random
import pickle
import glob

import atari_model


# Global Variables
NUM_SEEDS_TO_CHECK = 10
NUM_SEED_TRAILS = 30


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


def test_seed(seed_dict, config, m_id, env_seed, num_trails=NUM_SEED_TRAILS, monitor=True):
    """
    test if the model generalised or is only good on the specific
    env it stumbled upon. Creates a csv for each seed_dict and
    can make an mp4 for the best when monitor=True
    """

    new_folder = str(m_id)
    if os.path.isdir(new_folder):
        print('Folder already exists')
        return None
    else:
        os.makedirs(new_folder)

    our_rewards = []
    our_frames = []
    our_env_seeds = []

    seed = env_seed
    rsg = random_seed_generator(seed)

    model = atari_model.AtariModel(seed_dict, config)

    for n in range(num_trails):
        reward, frames, env_seed = model.evaluate_model(monitor=False,
                                                        set_env_seed=seed)
        seed = next(rsg)
        our_rewards.append(reward)
        our_frames.append(frames)
        our_env_seeds.append(env_seed)
        print('ID: ' + str(m_id) + ' run: ' + str(n), flush=True)

    csv_path = new_folder + '/test_seed_' + str(m_id) + '.csv'
    pd.DataFrame(
        {
            'tournament_env_seed': our_env_seeds,
            'reward': our_rewards,
            'frames': our_frames
        }
        ).to_csv(csv_path, index=False)
    
    # save boxplot of results
    ax = sns.boxplot(data=our_rewards, color="lightblue")
    ax = sns.swarmplot(data=our_rewards, color=".1")
    title = 'seed_' + str(config['run_seed'])
    ax.set_title(title)
    figure = ax.get_figure()
    plot_path = new_folder + '/' + title + '.png'
    figure.savefig(plot_path)

    # monitor best run
    if monitor:
        best_seed = our_env_seeds[np.argmax(our_rewards)]
        reward, frames, env_seed = model.evaluate_model(monitor=True,
                                                        set_env_seed=best_seed,
                                                        output_fn=new_folder)


def main():

    print("Does it generalise!?!?")

    # FOLDER_NAME = 'v/frostbite-experiment-1550464143.200036'

    # open json
    f_name = 'ga_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)

    # add time stamp to output_fname
    # config['output_fname'] = '.'

    # start timing
    # start = time.time()

    # set run seed and add generators to config
    # config['get_id'] = id_generator()
    config['get_seed'] = random_seed_generator(config['run_seed'])

    # move into directory
    os.chdir(config['output_fname'])

    # Load data (deserialize)
    files = glob.glob('tournament_winners*')
    tournament_winning_seed_dicts = {}
    for f in files:
        with open(f, 'rb') as handle:
            tournament_winning_seed_dicts.update(pickle.load(handle))

    # get results from run
    files = glob.glob('*results.csv')
    df_list = []
    for f in files:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list).sort_values('reward', ascending=False)

    # check generalisability for top __
    checked_ids = []
    for i in range(NUM_SEEDS_TO_CHECK+1):
        m_id = df.iloc[i]['id']
        if m_id in checked_ids:
            pass
        elif m_id in tournament_winning_seed_dicts.keys():
            checked_ids.append(m_id)
            test_seed(seed_dict=tournament_winning_seed_dicts[m_id],
                      config=config,
                      m_id=m_id,
                      env_seed=np.int(df.iloc[i]['tournament_env_seed']))
        else:
            print('ID not in seed dict')

    # create csv fpr top ids
    files = glob.glob('**/test_seed_*')
    df_list = []
    for f in files:
        temp_df = pd.read_csv(f)
        temp_df['id'] = f.split('.')[0]
        temp_df['run'] = False
        temp_df['run'][0] = True
        df_list.append(temp_df)
    df = pd.concat(df_list).sort_values('reward', ascending=False)
    csv_path = 'top_seeds.csv'
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
