import json
import time
import numpy as np
import matplotlib.pyplot as plt
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

    f_name = 'rs_frostbite.json'

    our_seeds = []
    our_rewards = []
    total_frames = 0

    with open(f_name) as f:
        info = json.load(f)

    # add time stamp to output_fname
    info['output_fname'] = info['output_fname'] + str(time.time())

    # start timing
    start = time.time()

    # start the 'random' search!
    for s in random_seed_generator(info['seed']):
        
        m = rs_model.RSModel(seed=s, info=info)
        reward, frames = m.evaluate_model()
        
        our_seeds.append(s)
        our_rewards.append(reward)
        total_frames += frames

        elapsed = (time.time() - start)
        print("Time: " + str(round(elapsed)) +
              ", Frames: " + str(total_frames), flush=True)

        if total_frames > info['max_frames']:
            break

    # get best seed
    print('recording best network', flush=True)
    best_seed = our_seeds[np.argmax(our_rewards)]
    m = rs_model.RSModel(seed=best_seed, info=info)
    _, _ = m.evaluate_model(monitor=True)

    # save our results
    # This will only work if the dir has been created above.
    csv_path = info['output_fname'] + '/results.csv'
    pd.DataFrame({'seed': our_seeds, 'reward': our_rewards}).to_csv(csv_path)

    print('all finished :D')


if __name__ == "__main__":
    main()
