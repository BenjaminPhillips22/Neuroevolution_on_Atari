import unittest
import torch
import copy
import random
import json

import base_model
import atari_model
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


class GATests(unittest.TestCase):

    def setUp(self):
        pass

    def test_the_test(self):
        '''
        testing the tester
        '''
        self.assertEqual(4*4, 16, 'something went wrong')

    def test_seed_dict(self):
        '''
        test that BigModel weights are the same and different
        as I would expect.
        '''
        # f_name = 'ga_frostbite.json'
        # with open(f_name) as f:
        #     config = json.load(f)

        # config['get_id'] = id_generator()
        # config['get_seed'] = random_seed_generator(config['run_seed'])

        # c1 = compressed_model.CompressedModel(config)

        # the_seeds = [x[0] for x in c1.seed_dict.values()]
        # first_seed = the_seeds[0]

        # for s in the_seeds:
        #     self.assertEqual(first_seed, s)

    def test_mutation(self):
            '''
            test mutation function is called properly within take_dna.
            Also, checks that id increments!
            '''
            # f_name = 'ga_frostbite.json'
            # with open(f_name) as f:
            #     config = json.load(f)

            # config['mutation_rate'] = 0.999

            # # generator_dict = {'id': id_generator(), 'seed': random_seed_generator()}
            # # set run seed and add generators to config
            # config['get_id'] = id_generator()
            # config['get_seed'] = random_seed_generator(config['run_seed'])
            # m1 = compressed_model.CompressedModel(config)

            # # generator_dict = {'id': id_generator(), 'seed': random_seed_generator()}
            # # set run seed and add generators to config
            # config['get_id'] = id_generator()
            # config['get_seed'] = random_seed_generator(config['run_seed'])
            # m2 = compressed_model.CompressedModel(config)
            
            # self.assertEqual(m1.seed_dict, m2.seed_dict)

            # m1_initial_seed_dict = copy.deepcopy(m1.seed_dict)
            
            # # set seed for 'take_dna'
            # random.seed(22)

            # m1.take_dna(m2, config)

            # # print(m1_initial_seed_dict)
            # # print(m1.seed_dict)

            # self.assertEqual(m1_initial_seed_dict['conv1.weight'][0], m1.seed_dict['conv1.weight'][0])
            # self.assertEqual(len(m1.seed_dict['conv1.weight']), 2)

            # self.assertEqual(m1.id, 2)


if __name__ == '__main__':
    unittest.main()
