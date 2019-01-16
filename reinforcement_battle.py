#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for command line battle between agents on a game
"""


import argparse
# from itertools import product
import json
import os

import reinforcement_ecosystem.agents as agents
import reinforcement_ecosystem.games as games


# list_of_agents_couple = set(product((el for el in globals().keys() if 'Agent' in el), repeat=2))
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
    if not os.path.exists('./tf_logs'):
        os.mkdir('./tf_logs', 0o755)
    if not os.path.exists('./csv_logs'):
        os.mkdir('./csv_logs', 0o755)
    args = vars(parser.parse_args())
    agent1 = getattr(agents, f"{args['agent1']}Agent")
    agent1_args = json.loads(args['agent1_args']) if args['agent1_args'] else {}
    agent2 = getattr(agents, f"{args['agent2']}Agent")
    agent2_args = json.loads(args['agent2_args']) if args['agent2_args'] else {}
    game_runner = getattr(games, f"{args['game']}Runner")
    game_state = getattr(games, f"{args['game']}GameState")
    gs_args = json.loads(args['gs_args']) if args['gs_args'] else {}
    agent1_log_args = '_'.join(f'{k}-{v}' for k, v in agent1_args.items())
    agent2_log_args = '_'.join(f'{k}-{v}' for k, v in agent2_args.items())
    log_name = f"{args['game']}__{args['agent1']}_VS_{args['agent2']}__{args['max_rounds']}"\
        f"__[{agent1_log_args}]__[{agent2_log_args}]"
    if args['no_gpu']:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print(args['agent1'], 'VS', args['agent2'], 'on', args['game'], 'for', str(args['max_rounds']), 'rounds')
    csv_data = {
        'game': args['game'],
        'battle_name': f"{args['agent1']}_VS_{args['agent2']}",
        'num_of_games': args['max_rounds'],
        'agent1_name': args['agent1'],
        'round_number': 0,
        'agent1_mean_action_duration_sum': 0,
        'agent1_mean_accumulated_reward_sum': 0,
        'agent2_name': args['agent2'],
        'agent2_mean_action_duration_sum': 0,
        'agent2_mean_accumulated_reward_sum': 0,
        'mean_run_duration': 0
    }
    game_runner(agent1(**agent1_args), agent2(**agent2_args), csv_data, log_name)\
        .run(game_state(**gs_args), args['max_rounds'])
