#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining GameRunners
"""


import tensorflow as tf

from reinforcement_ecosystem.config import PRINT_EACH
from .agent import Agent
from .game_state import GameState


class GameRunner:
    """
    GameRunner base class
    """

    def __init__(self, agent1: Agent, agent2: Agent, tf_log_dir: str) -> None:
        """
        Initializer for Game Runner
        :param agent1: The first player agent
        :param agent2: The second player agent
        :param tf_log_dir: Where Tensorflow should log
        """
        self.agents = agent1, agent2
        self.writer = tf.summary.FileWriter(tf_log_dir)

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
            self.writer.add_summary(tf.Summary(value=value_summary), episode_id)
            episode_id += 1


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
