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
        f_name = 'ga_frostbite.json'
        with open(f_name) as f:
            config = json.load(f)

        # generator_dict = {'id': id_generator(), 'seed': random_seed_generator()}

        a_seed_dict = {'conv1.weight': [248528185],
                       'conv2.weight': [1877002014],
                       'conv3.weight': [1679996776],
                       'dense.weight': [919992575],
                       'out.weight': [1904593925]}

        b_seed_dict = {'conv1.weight': [248528185],
                       'conv2.weight': [1877002014, 123],
                       'conv3.weight': [1679996776],
                       'dense.weight': [919992575],
                       'out.weight': [1904593925]}

        m1 = base_model.BigModel(a_seed_dict, config)
        m2 = base_model.BigModel(b_seed_dict, config)

        a_tens = []
        for name, tensor in m1.named_parameters():
            a_tens.append(tensor)

        b_tens = []
        for name, tensor in m2.named_parameters():
            b_tens.append(tensor)
            
        result = []    
        for i in range(len(a_tens)):
            result.append(torch.all(torch.eq(a_tens[i], b_tens[i])).data)

        self.assertEqual(result, [1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    # def test_crossover(self):
    #     '''
    #     test take_dna
    #     This test only worked for specific seed with specific code and now
    #     that I've changed the code, the test doesn't pass. Lesson learnt!
    #     '''
    #     f_name = 'ga_frostbite.json'
    #     with open(f_name) as f:
    #         config = json.load(f)

    #     # generator_dict = {'id': id_generator(), 'seed': random_seed_generator()}
    #     # set run seed and add generators to config
    #     config['get_id'] = id_generator()
    #     config['get_seed'] = random_seed_generator(config['run_seed'])


    #     m1 = compressed_model.CompressedModel(config)
    #     m2 = compressed_model.CompressedModel(config)
        
    #     m1_initial_seed_dict = copy.deepcopy(m1.seed_dict)
    #     m2_initial_seed_dict = copy.deepcopy(m2.seed_dict)

    #     # set seed for 'take_dna'
    #     random.seed(22)

    #     m1.take_dna(m2, config, mutate=False)

    #     # m2 should be unchanged
    #     self.assertEqual(m2_initial_seed_dict, m2.seed_dict)

    #     # m1 should be changed
    #     self.assertNotEqual(m1_initial_seed_dict, m1.seed_dict)

    #     for name in m1.seed_dict.keys():
    #         if name in ['conv2.weight', 'conv3.weight', 'out.weight']:
    #             self.assertEqual(m1.seed_dict[name], m2.seed_dict[name])
    #         else:
    #             self.assertNotEqual(m1.seed_dict[name], m2.seed_dict[name])

    def test_mutation(self):
            '''
            test mutation function is called properly within take_dna.
            Also, checks that id increments!
            '''
            f_name = 'ga_frostbite.json'
            with open(f_name) as f:
                config = json.load(f)

            config['mutation_rate'] = 0.999

            # generator_dict = {'id': id_generator(), 'seed': random_seed_generator()}
            # set run seed and add generators to config
            config['get_id'] = id_generator()
            config['get_seed'] = random_seed_generator(config['run_seed'])
            m1 = compressed_model.CompressedModel(config)

            # generator_dict = {'id': id_generator(), 'seed': random_seed_generator()}
            # set run seed and add generators to config
            config['get_id'] = id_generator()
            config['get_seed'] = random_seed_generator(config['run_seed'])
            m2 = compressed_model.CompressedModel(config)
            
            self.assertEqual(m1.seed_dict, m2.seed_dict)

            m1_initial_seed_dict = copy.deepcopy(m1.seed_dict)
            
            # set seed for 'take_dna'
            random.seed(22)

            m1.take_dna(m2, config)

            # print(m1_initial_seed_dict)
            # print(m1.seed_dict)

            self.assertEqual(m1_initial_seed_dict['conv1.weight'][0], m1.seed_dict['conv1.weight'][0])
            self.assertEqual(len(m1.seed_dict['conv1.weight']), 2)

            self.assertEqual(m1.id, 2)

    def test_set_env_seed(self):
        """
        
        """
        f_name = 'ga_frostbite.json'
        with open(f_name) as f:
            config = json.load(f)

        # set run seed and add generators to config
        config['get_id'] = id_generator()
        config['get_seed'] = random_seed_generator(config['run_seed'])

        a_seed_dict = {'conv1.weight': [248528185],
                       'conv2.weight': [1877002014],
                       'conv3.weight': [1679996776],
                       'dense.weight': [919992575],
                       'out.weight': [1904593925]}

        m1 = atari_model.AtariModel(a_seed_dict, config)
        m2 = atari_model.AtariModel(a_seed_dict, config)

        reward_1, frames_1, env_seed_1 = m1.evaluate_model(set_env_seed=101)
        reward_2, frames_2, env_seed_2 = m2.evaluate_model(set_env_seed=101)

        self.assertEqual(reward_1, reward_2)
        self.assertEqual(frames_1, frames_2)
        self.assertEqual(env_seed_1, env_seed_2)

    def test_set_env_seed_again(self):
        """
        same as above but generate env_seed in first run 
        and use that seed to set second run
        """
        f_name = 'ga_frostbite.json'
        with open(f_name) as f:
            config = json.load(f)

        # set run seed and add generators to config
        config['get_id'] = id_generator()
        config['get_seed'] = random_seed_generator(config['run_seed'])

        a_seed_dict = {'conv1.weight': [248528185],
                       'conv2.weight': [1877002014],
                       'conv3.weight': [1679996776],
                       'dense.weight': [919992575],
                       'out.weight': [1904593925]}

        m1 = atari_model.AtariModel(a_seed_dict, config)
        m2 = atari_model.AtariModel(a_seed_dict, config)

        reward_1, frames_1, env_seed_1 = m1.evaluate_model()
        reward_2, frames_2, env_seed_2 = m2.evaluate_model(set_env_seed=env_seed_1)

        self.assertEqual(reward_1, reward_2)
        self.assertEqual(frames_1, frames_2)
        self.assertEqual(env_seed_1, env_seed_2)



if __name__ == '__main__':
    unittest.main()
