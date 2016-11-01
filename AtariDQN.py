#!/usr/bin/env python
import os
import argparse

import cv2
from game.Atari import Atari
from BrainDQN_Nature import BrainDQN
import numpy as np

# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))

def playAtari(args):
    # Step 1: init BrainDQN
    # Step 2: init Flappy Bird Game
    atari = Atari('breakout.bin')
    actions = len(atari.legal_actions)
    brain = BrainDQN(actions, args.checkpoints, args.summary)

    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1, 0, 0, 0])  # do nothing
    observation0, reward0, terminal = atari.next(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation0 = observation0[26:110, :]
    _, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while True:
        action = brain.getAction()
        nextObservation, reward, terminal = atari.next(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal)

def main():
    parser = argparse.ArgumentParser()
    data_basepath = 'data'
    parser.add_argument('-c', '--checkpoints', dest='checkpoints', default=os.path.join(data_basepath, 'checkpoints'),
                        help='Path where to store checkpoints (i.e partial training)')
    parser.add_argument('-s', '--summary', dest='summary', default=os.path.join(data_basepath, 'summary'),
                        help='Path where to store summary data (for tensorboard)')
    args = parser.parse_args()

    playAtari(args)

if __name__ == '__main__':
    main()
