#!/usr/bin/env python
import os
import argparse

import cv2
from game.Atari import Atari
from BrainDQN_Nature import BrainDQN
import numpy as np

# _preprocess raw image to 80*80 gray image
def _preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))

def _play_atari(args):
    atari = Atari('breakout.bin')

    action0 = np.array([1, 0, 0, 0])  # do nothing
    observation0, _, terminal = atari.next(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation0 = observation0[26:110, :]
    _, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)

    brain = BrainDQN(observation0, len(atari.legal_actions), args.checkpoints, args.summary)

    while True:
        action = brain.get_action()
        next_observation, reward, terminal = atari.next(action)
        next_observation = _preprocess(next_observation)
        brain.set_perception(next_observation, action, reward, terminal)

def main():
    parser = argparse.ArgumentParser()
    data_basepath = 'data'
    parser.add_argument('-c', '--checkpoints', dest='checkpoints', default=os.path.join(data_basepath, 'checkpoints'),
                        help='Path where to store checkpoints (i.e partial training)')
    parser.add_argument('-s', '--summary', dest='summary', default=os.path.join(data_basepath, 'summary'),
                        help='Path where to store summary data (for tensorboard)')
    args = parser.parse_args()

    _play_atari(args)

if __name__ == '__main__':
    main()
