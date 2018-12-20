#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining GameRunners
"""


from csv import DictWriter

import tensorflow as tf

from reinforcement_ecosystem.config import *
from .agent import Agent
from .game_state import GameState


class GameRunner:
    """
    GameRunner base class
    """

    def __init__(self, agent1: Agent, agent2: Agent, csv_data: dict, log_name: str) -> None:
        """
        Initializer for Game Runner
        :param agent1: The first player agent
        :param agent2: The second player agent
        :param csv_data: The CSV data to use as logging
        :param log_name: The name of the logs
        """
        self.agents = agent1, agent2
        self.csv_writer = open('{}/{}.csv'.format(CSV_LOG_DIR, log_name), 'w')
        self.tf_writer = tf.summary.FileWriter('{}/{}'.format(TF_LOG_DIR, log_name))
        self.csv_data = csv_data

    def _run(self, initial_game_state: GameState) -> dict:
        raise NotImplementedError()

    def run(self, initial_game_state: GameState, max_rounds: int = -1) -> None:
        """
        Run the game with a specific `GameState`
        :param initial_game_state: The initial `GameState` to use for running the game
        :param max_rounds: The number maximum of rounds to play the game
        """
        episode_id = 0
        print_every = int(max_rounds * PRINT_EACH)
        csv_log_every = int(max_rounds * CSV_LOG_EACH)
        dw = DictWriter(self.csv_writer, fieldnames=self.csv_data.keys())
        dw.writeheader()
        for mr in range(max_rounds):
            if mr % print_every == 0:
                print('Round N :', str(mr))
            stats = self._run(initial_game_state)
            value_summary = [
                    tf.Summary.Value(tag='agent1_action_mean_duration',
                                     simple_value=stats['mean_action_duration_sum_a1'] / stats['round_step']),
                    tf.Summary.Value(tag='agent2_action_mean_duration',
                                     simple_value=stats['mean_action_duration_sum_a2'] / stats['round_step']),
                    tf.Summary.Value(tag='agent1_accumulated_reward',
                                     simple_value=stats['mean_accumulated_reward_sum_a1']),
                    tf.Summary.Value(tag='agent2_accumulated_reward',
                                     simple_value=stats['mean_accumulated_reward_sum_a2'])
                ]
            self.tf_writer.add_summary(tf.Summary(value=value_summary), episode_id)
            if mr % csv_log_every == 0:
                self.csv_data['round_number'] = episode_id
                self.csv_data['agent1_mean_action_duration_sum'] = stats['mean_action_duration_sum_a1']
                self.csv_data['agent1_mean_accumulated_reward_sum'] = stats['mean_accumulated_reward_sum_a1']
                self.csv_data['agent2_mean_action_duration_sum'] = stats['mean_action_duration_sum_a2']
                self.csv_data['agent2_mean_accumulated_reward_sum'] = stats['mean_accumulated_reward_sum_a2']
                dw.writerow(self.csv_data)
            episode_id += 1

    def __del__(self) -> None:
        """
        Actions to do when the object is cleanup by the VM
        """
        self.csv_writer.close()

    # @classmethod
    # def runner_helper(cls, game: str, num_games: int, war_name: str, agent1_name: str,
    #                   agent1: 'Agent', agent2_name: str, agent2: 'Agent', stats_csv: 'IO', **kwargs) -> None:
    #     print(war_name, str(num_games), 'runs')
    #     score = cls(agent1, agent2, **kwargs).run(num_games)
    #     game_stats = OrderedDict({
    #         'game': game,
    #         'battle_name': war_name,
    #         'num_of_games': num_games,
    #         'agent1_name': agent1_name,
    #         'agent1_nb_victory': score[0],
    #         'agent1_victory_rate': (score[0] / num_games) * 100,
    #         'agent2_name': agent2_name,
    #         'agent2_nb_victory': score[1],
    #         'agent2_victory_rate': (score[1] / num_games) * 100,
    #         'draw_nb': score[2],
    #         'draw_rate': (score[2] / num_games) * 100
    #     })
    #     pprint(game_stats)
    #     dw = csv.DictWriter(stats_csv, fieldnames=game_stats.keys())
    #     dw.writerow(game_stats)
