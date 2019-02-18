"""
Continue the genetic algorithm from where it left off.
Will have to update if dealing with new tournament_winners pickle....
"""


import os
import json
import time
import numpy as np
import pandas as pd
import random
import pickle

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

    # folder where we want to continue from
    FOLDER_NAME = 'v/frostbite-experiment-1550467378.821758'

    our_env_seeds = []
    our_rewards = []
    our_time = []
    our_frames = []
    our_ids = []
    our_tournament_number = []
    total_frames = 0

    # open json
    f_name = 'ga_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)

    # check tournament size isn't bigger than population
    assert config['tournament_size'] <= config['population_size']

    # add time stamp to output_fname
    config['output_fname'] = FOLDER_NAME

    # creating a new folder
    # os.makedirs(config['output_fname'])

    # save config
    # csv_path = config['output_fname'] + '/config.csv'
    # pd.DataFrame.from_dict(config, orient='index').to_csv(csv_path)

    # start timing
    start = time.time()

    # get saved population and seed
    # Load data (deserialize)
    print("Loading Pickled Generation")
    path = FOLDER_NAME + "/pop_and_seed.pickle"
    with open(path, 'rb') as handle:
        pop_and_seed = pickle.load(handle)

    # create the population
    population = pop_and_seed['population']

    # get tournament number
    tournament_number = pop_and_seed['tournament']

    # set run seed and add generators to config
    config['get_id'] = id_generator(pop_and_seed['id'])
    config['get_seed'] = random_seed_generator(pop_and_seed['seed'])

    # dict to store tournament winners
    tournament_winning_seed_dicts = {}

    # start the 'GA' search!
    while total_frames < config['max_frames']:

        # tournament lists
        tournament_number += 1
        tournament_env_seed = []
        tournament_rewards = []
        tournament_time = []
        tournament_frames = []
        tournament_ids = []
        tournament_tournament_number = []

        # select individuals for tournament
        random.seed(next(config['get_seed']))
        tournament_indices = random.sample(population=range(config['population_size']), k=config['tournament_size'])

        # get rewards
        for i in tournament_indices:

            reward, frames, env_seed = population[i].evaluate_compressed_model(config)
            tournament_rewards.append(reward)
            tournament_frames.append(frames)
            tournament_env_seed.append(env_seed)

            # update total_frames
            total_frames += frames

            tournament_ids.append(population[i].id)
            tournament_tournament_number.append(tournament_number)

            # record time
            elapsed = (time.time() - start)
            tournament_time.append(elapsed)

            print("Time: " + str(round(elapsed)) +
                  ", Frames: " + str(total_frames) +
                  ", id: " + str(population[i].id) +
                  ", reward: " + str(reward) +
                  ", percent: " + str(round(total_frames*100/config["max_frames"], 2)), flush=True)

        # Who won the tournament?
        tournament_index_of_max_reward = np.argmax(tournament_rewards)
        print("TOURNAMENT NUMBER " + str(tournament_number) +
              ", winning id: " + str(tournament_ids[tournament_index_of_max_reward]) +
              ", reward: " + str(tournament_rewards[tournament_index_of_max_reward]), flush=True)

        # save the winner
        tournament_indices_winner = tournament_indices[tournament_index_of_max_reward]
        tournament_winning_seed_dicts[population[tournament_indices_winner].id] = population[tournament_indices_winner].seed_dict

        # the not-winners take the dna of the winner
        for i in tournament_indices:
            if i == tournament_indices_winner:
                pass
            else:
                population[i].take_dna(population[tournament_indices_winner], config)

        # update our lists
        our_time += tournament_time
        our_env_seeds += tournament_env_seed
        our_rewards += tournament_rewards
        our_frames += tournament_frames
        our_ids += tournament_ids
        our_tournament_number += tournament_tournament_number

        if tournament_number % 10 == 0:

            csv_path = config['output_fname'] + "/tournament_number_" + str(tournament_number) + '_results.csv'

            pd.DataFrame(
                {
                    'id': our_ids,
                    'tournament_env_seed': our_env_seeds,
                    'tournament_number': our_tournament_number,
                    'reward': our_rewards,
                    'time': our_time,
                    'frames': our_frames
                }
                ).to_csv(csv_path, index=False)
            our_env_seeds = []
            our_rewards = []
            our_time = []
            our_frames = []
            our_ids = []
            our_tournament_number = []

            # save best compressed_model seed_dicts
            # no need to create a new pickle each time, replace old with updated new
            pickle_path = config['output_fname'] + "/tournament_winners_" + str(pop_and_seed['tournament']) + "_.pickle"
            with open(pickle_path, 'wb') as handle:
                pickle.dump(tournament_winning_seed_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # save the latest generation in case I want to re-run from there.
            # again, replace previous with update.
            pickle_path = config['output_fname'] + "/pop_and_seed_" + str(pop_and_seed['tournament']) + "_.pickle"
            with open(pickle_path, 'wb') as handle:
                pickle.dump(
                    {
                        'population': population,
                        'seed': next(config['get_seed']),
                        'tournament': tournament_number,
                        'id': next(config['get_id'])
                    },
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save any last our results
    if len(our_rewards) > 0:
        csv_path = config['output_fname'] + "/tournament_number_" + str(tournament_number) + '_results.csv'

        pd.DataFrame(
            {
                'id': our_ids,
                'tournament_env_seed': our_env_seeds,
                'tournament_number': our_tournament_number,
                'reward': our_rewards,
                'time': our_time,
                'frames': our_frames
            }
            ).to_csv(csv_path, index=False)

        # save best compressed_model seed_dicts
        # no need to create a new pickle each time, replace old with updated new
        pickle_path = config['output_fname'] + "/tournament_winners_" + str(pop_and_seed['tournament']) + "_.pickle"
        with open(pickle_path, 'wb') as handle:
            pickle.dump(tournament_winning_seed_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save the latest generation in case I want to re-run from there.
        # again, replace previous with update.
        pickle_path = config['output_fname'] + "/pop_and_seed_" + str(pop_and_seed['tournament']) + "_.pickle"
        with open(pickle_path, 'wb') as handle:
            pickle.dump(
                {
                    'population': population,
                    'seed': next(config['get_seed']),
                    'tournament': tournament_number,
                    'id': next(config['get_id'])
                },
                handle, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = (time.time() - start)
    print("Time: " + str(round(elapsed)))

    print('all finished :D')


if __name__ == "__main__":
    main()
