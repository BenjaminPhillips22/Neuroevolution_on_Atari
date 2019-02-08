"""
Run the genetic algorithm
"""

import os
import json
import time
import numpy as np
import pandas as pd
import random

import compressed_model


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


def main():

    # our_seeds = []
    our_rewards = []
    our_time = []
    our_frames = []
    our_ids = []
    total_frames = 0
    tournament_number = 0

    # open json
    f_name = 'ga_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)

    # check tournament size isn't bigger than population
    assert config['tournament_size'] <= config['population_size']

    # add time stamp to output_fname
    config['output_fname'] = config['output_fname'] + str(time.time())

    # creating a new folder
    os.makedirs(config['output_fname'])

    # save config
    csv_path = config['output_fname'] + '/config.csv'
    pd.DataFrame.from_dict(config, orient='index').to_csv(csv_path)

    # start timing
    start = time.time()

    # set run seed and create generators
    generator_dict = {'id': id_generator(config['run_seed']), 'seed': random_seed_generator()}

    # create the population
    population = [compressed_model.CompressedModel(generator_dict, config) for _ in range(config["population_size"])]

    # start the 'GA' search!
    while total_frames < config['max_frames']:

        # tournament lists
        tournament_number += 1
        tournament_rewards = []
        tournament_time = []
        tournament_frames = []
        tournament_ids = []
        tournament_winning_seed_dicts = {}

        # select individuals for tournament
        random.seed(next(config['seed']))
        tournament_indices = random.sample(population=range(config['population_size']), k=config['tournament_size'])

        # get rewards
        for i in tournament_indices:
            
            reward, frames = population[i].evaluate_compressed_model()
            tournament_rewards.append(reward)
            tournament_frames.append(frames)
            
            # update total_frames
            total_frames += frames

            tournament_ids.append(population[i].id)
            
            # record time
            elapsed = (time.time() - start)
            tournament_time.append(elapsed)

            print("Time: " + str(round(elapsed)) +
                  ", Frames: " + str(total_frames) +
                  ", id: " + str(population[i].id),
                  ", percent: " + str(round(total_frames*100/config["max_frames"], 2)), flush=True)

        # Who won the tournament?
        tournament_index_of_max_reward = np.argmax(tournament_rewards)
        print("TOURNAMENT WINNER id: " +
              str(tournament_ids[tournament_index_of_max_reward]) +
              " reward: " + tournament_rewards[tournament_index_of_max_reward])

        # save the winner
        tournament_indices_winner = tournament_indices[tournament_index_of_max_reward]
        tournament_winning_seed_dicts[population[tournament_indices_winner].id] = population[tournament_indices_winner].seed_dict
        
        # the not-winners take the dna of the winner
        for i in tournament_indices:
            if i == tournament_indices_winner:
                pass
            else:
                population[i].take_dna(population[tournament_indices_winner])

        # update our lists
        our_time += tournament_time
        our_rewards += tournament_rewards
        our_frames += tournament_frames
        our_ids += tournament_ids

        if tournament_number % 10 == 0:

            csv_path = config['output_fname'] + "/tournament_number_" + str(tournament_number) + '_results.csv'

            pd.DataFrame(
                {
                    'id': our_ids,
                    'reward': our_rewards,
                    'time': our_time,
                    'frames': our_frames
                }
                ).to_csv(csv_path, index=False)
            our_rewards = []
            our_time = []
            our_frames = []
            our_ids = []

            # save best compressed_model seed_dicts
            # no need to create a new pickle each time, replace old with updated new
            pickle_path = config['output_fname'] + "/tournament_winners.pickle"
            with open(pickle_path, 'wb') as handle:
                pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Load data (deserialize)
            # with open('filename.pickle', 'rb') as handle:
            #     unserialized_data = pickle.load(handle)


    # save any last our results
    if len(our_rewards) > 0:
            csv_path = config['output_fname'] + "/tournament_number_" +
            str(tournament_number) + '_results.csv'

            pd.DataFrame(
                {
                    'id': our_ids,
                    'reward': our_rewards,
                    'time': our_time,
                    'frames': our_frames
                }
                ).to_csv(csv_path, index=False)
    elapsed = (time.time() - start)

    # save best compressed_model seed_dicts
    # no need to create a new pickle each time, replace old with updated new
    pickle_path = config['output_fname'] + "/tournament_winners.pickle"
    with open(pickle_path, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Time: " + str(round(elapsed)))

    print('all finished :D')


if __name__ == "__main__":
    main()
