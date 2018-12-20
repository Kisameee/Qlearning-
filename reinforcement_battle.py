#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for command line battle between agents on a game
"""


import os
import argparse
import json

# import reinforcement_ecosystem.agents as agents
# import reinforcement_ecosystem.games as games
from reinforcement_ecosystem.agents import *
from reinforcement_ecosystem.games import *
from reinforcement_ecosystem.config import TF_LOG_DIR


parser = argparse.ArgumentParser(description='Command line tool for battling two agents on a game')
parser.add_argument('agent1', type=str, help='The Player 1 Agent')
parser.add_argument('agent2', type=str, help='The Player 2 Agent')
parser.add_argument('game', type=str, help='The game to play on')
parser.add_argument('max_rounds', type=int, help='The maximum number of rounds to play')
parser.add_argument('--no-gpu', help='If the GPU should be used or not', action='store_true', default=False)
parser.add_argument('--agent1_args', type=str, help='Agent 1 arguments', default='')
parser.add_argument('--agent2_args', type=str, help='Agent 2 arguments', default='')
parser.add_argument('--gs_args', type=str, help='GameState arguments', default='')


if __name__ == '__main__':
    args = vars(parser.parse_args())
    agent1 = globals()['{}Agent'.format(args['agent1'])]  # try getattr ?
    agent1_args = json.loads(args['agent1_args']) if args['agent1_args'] else {}
    agent2 = globals()['{}Agent'.format(args['agent2'])]  # try getattr ?
    agent2_args = json.loads(args['agent2_args']) if args['agent2_args'] else {}
    game_runner = globals()['{}Runner'.format(args['game'])]  # try getattr ?
    game_state = globals()['{}GameState'.format(args['game'])]  # try getattr ?
    gs_args = json.loads(args['gs_args']) if args['gs_args'] else {}
    log_dir = '{}/{}__{}_VS_{}__{}'.format(TF_LOG_DIR, args['game'], args['agent1'],
                                           args['agent2'], str(args['max_rounds']))
    if args['no_gpu']:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(args['agent1'], 'VS', args['agent2'], 'on', args['game'], 'for', str(args['max_rounds']), 'rounds')
    game_runner(agent1(**agent1_args), agent2(**agent2_args), log_dir)\
        .run(game_state(**gs_args), args['max_rounds'])
