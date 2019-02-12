import os
import json
import time
import numpy as np
import pandas as pd
import random
import pickle

import atari_model


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

    seed_dict = 

    config = 

    model = atari_model.AtariModel(seed_dict, config)
    reward, frames = model.evaluate_model()
    return reward, frames