# In GA folder

import torch
import random

import atari_model


class CompressedModel():
    """
    Used to evaluate the seed_dict and make babies
    """
    def __init__(self, gen_dict, config):

        self.config = config
        self.gen_seed = gen_dict['seed']
        self.gen_id = gen_dict['id']

        self.id = next(self.gen_id)

        self.mutation_rate = 1/24
        self.seed_dict = {}

        layer_names = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'dense.weight', 'out.weight']
        for name in layer_names:
            self.seed_dict[name] = [next(self.gen_seed)]

    def mutate(self):
        """
        Given a small chance, add a random seed to a list of seeds
        """
        for name, _ in self.seed_dict.items():
            if random.random() < self.mutation_rate:
                self.seed_dict[name].append(next(self.gen_seed))

    def take_dna(self, OtherCompressedModel):
        """
        Perform crossover to create a child and update this
        CompressedModel's seed_dict
        """

    def evaluate_compressed_model(self):
        """
        make AtariModel, get reward and frames from it.
        """
        model = atari_model.AtariModel(self.seed_dict, self.config)
        reward, frames = model.evaluate_model()
        return reward, frames
