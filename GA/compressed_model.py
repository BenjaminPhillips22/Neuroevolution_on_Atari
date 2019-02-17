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
        self.seed_dict = {}

        layer_names = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'dense.weight', 'out.weight']
        for name in layer_names:
            self.seed_dict[name] = [next(config['get_seed'])]

    def mutate(self, config):
        """
        Given a small chance, add a random seed to a list of seeds
        """
        for name, _ in self.seed_dict.items():
            random.seed(next(config['get_seed']))
            if random.random() < self.mutation_rate:
                self.seed_dict[name].append(next(self.get_seed))

    def take_dna(self, OtherCompressedModel, config, mutate=True):
        """
        Perform crossover to create a child and update this
        CompressedModel's seed_dict
        """
        random.seed(next(config['get_seed']))
        for name in self.seed_dict.keys():
            if random.random() < 0.5:
                self.seed_dict[name] = copy.deepcopy(OtherCompressedModel.seed_dict[name])

        if mutate:
            self.mutate(config)

        # update id
        self.id = next(self.gen_id)

    def evaluate_compressed_model(self, config):
        """
        make AtariModel, get reward and frames from it.
        """
        model = atari_model.AtariModel(self.seed_dict, config)
        reward, frames, env_seed = model.evaluate_model()
        return reward, frames, env_seed
