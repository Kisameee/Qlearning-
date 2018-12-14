#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Init file for the agents package
"""


from .moimcts import *
from .commandline import CommandLineAgent
from.qlearning import TabularQLearningAgent, DeepQLearningAgent
from .random import RandomAgent, RandomRolloutAgent
from .ppo import PPOWithMultipleTrajectoriesMultiOutputsAgent
from .reinforce import ReinforceClassicAgent, ReinforceClassicWithMultipleTrajectoriesAgent
