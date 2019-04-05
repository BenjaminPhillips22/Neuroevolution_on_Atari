# In GA folder

import random
import copy

import atari_model


class CompressedModel():
    """
    Used to evaluate the seed_dict and make babies
    """
    def __init__(self, config):

        # self.config = config
        # self.get_seed = config['get_seed']
        # self.gen_id = config['get_id']

        self.id = next(config['get_id'])

        self.mutation_rate = config['mutation_rate']
        first_seed = next(config['get_seed'])
        self.seed_list = [first_seed]

    def mutate(self, config):
        """
        when called, adds the same extra seed to each layer
        """
        new_seed = next(config['get_seed'])
        self.seed_list.append(new_seed)

    def take_dna(self, OtherCompressedModel, config, mutate=True):
        """
        This individual is replaced with the winner, mutate if mutate=True.
        """
        self.seed_list = copy.deepcopy(OtherCompressedModel.seed_list)

        if mutate:
            self.mutate(config)

        # update id
        self.id = next(config['get_id'])

    def evaluate_compressed_model(self, config):
        """
        make AtariModel, get reward and frames from it.
        """
        # create seed_dict that AtariModel can use
        layer_names = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'dense.weight', 'out.weight']
        temp_seed_dict = {}
        for name in layer_names:
            temp_seed_dict[name] = self.seed_list

        model = atari_model.AtariModel(temp_seed_dict, config)
        reward, frames, env_seed = model.evaluate_model()
        return reward, frames, env_seed
