import os
import json
import time
import numpy as np
import pandas as pd
import random

import rs_model


def random_seed_generator(seed=2):
    while True:
        random.seed(seed)
        new_rand = random.randint(0, 2**31-1)
        yield new_rand
        seed = new_rand


def main():

    our_seeds = []
    our_rewards = []
    our_time = []
    our_frames = []
    our_ids = []
    total_frames = 0

    # open json
    f_name = 'rs_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)

    # add time stamp to output_fname
    config['output_fname'] = config['output_fname'] + str(time.time())

    # check if v/ exists, if not, create it
    # if not os.path.exists('/v'):
    #     os.mkdir('/v')
    #     print('v directory created')

    # creating a new folder
    os.makedirs(config['output_fname'])

    # save config
    csv_path = config['output_fname'] + '/config.csv'
    pd.DataFrame.from_dict(config, orient='index').to_csv(csv_path)

    # start timing
    start = time.time()

    # set run seed
    get_seed = random_seed_generator(config['run_seed'])

    # start the 'random' search!
    i = 0
    while total_frames < config['max_frames']:

        i += 1
        s = next(get_seed)
        m = rs_model.RSModel(seed=s, config=config)
        reward, frames = m.evaluate_model()

        elapsed = (time.time() - start)

        our_time.append(elapsed)
        our_seeds.append(s)
        our_rewards.append(reward)
        our_frames.append(frames)
        our_ids.append(i)
        total_frames += frames

        print("Time: " + str(round(elapsed)) +
              ", Frames: " + str(total_frames) +
              ", id: " + str(i),
              ", seed: " + str(s) + 
              ", percent: " + str(round(total_frames*100/config["max_frames"], 2)), flush=True)

        if i % 10 == 0:
            max_r_index = np.argmax(our_rewards)
            print('max reward: ', str(our_rewards[max_r_index]), ' seed: ', str(our_seeds[max_r_index]))
        
        if i % 10 == 0:
            csv_path = config['output_fname'] + "/id_" + str(i) + '_results.csv'
            pd.DataFrame(
                {                
                    'id': our_ids,
                    'seed': our_seeds,
                    'reward': our_rewards,
                    'time': our_time,
                    'frames': our_frames
                }
                ).to_csv(csv_path, index=False)
            our_seeds = []
            our_rewards = []
            our_time = []
            our_frames = []
            our_ids = []

    # save any last our results
    if len(our_seeds) > 0:
        csv_path = config['output_fname'] + "/id_" + str(i) + '_results.csv'
        pd.DataFrame(
            {
                'id': our_ids,
                'seed': our_seeds,
                'reward': our_rewards,
                'time': our_time,
                'frames': our_frames
            }
            ).to_csv(csv_path, index=False)
    elapsed = (time.time() - start)
    print("Time: " + str(round(elapsed)))

    print('all finished :D')


if __name__ == "__main__":
    main()
