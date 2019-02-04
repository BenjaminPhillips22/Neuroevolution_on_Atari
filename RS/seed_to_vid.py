import os
import time
import json
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt

import gym
from gym import wrappers

import rs_model


def main():

    # start timing
    start = time.time()

    # open json
    f_name = 'rs_frostbite.json'
    with open(f_name) as f:
        config = json.load(f)
    
    # add time to folder name
    config['output_fname'] = config['output_fname'] + 's2v-' + str(time.time())

    # creating a new folder
    os.makedirs(config['output_fname'])

    # get best seed
    print('recording best network', flush=True)
    best_seed = 1098095577
    m = rs_model.RSModel(seed=best_seed, config=config)
    r, f = m.evaluate_model(monitor=True)

    elapsed = (time.time() - start)
    print("Time: " + str(round(elapsed)))

    # save our results
    csv_path = config['output_fname'] + '/results.csv'
    pd.DataFrame(
        {
            'seed': [best_seed],
            'reward': [r],
            'time': [elapsed],
            'frames': [f]
        }
        ).to_csv(csv_path)


if __name__ == "__main__":
    main()
