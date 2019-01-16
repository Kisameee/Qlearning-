#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining GameRunners
"""


from csv import DictWriter
from typing import Tuple

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
        self.csv_writer = open(f'{CSV_LOG_DIR}/{log_name}.csv', 'w')
        self.tf_writer = tf.summary.FileWriter(f'{TF_LOG_DIR}/{log_name}')
        self.csv_data = csv_data

    def _run(self, initial_game_state: GameState) -> dict:
        """
        Run the game with a specific `GameState` for a round
        :param initial_game_state: The gamestate to play on
        :return: The dict of stats of the game
        """
        raise NotImplementedError

    def run(self, initial_game_state: GameState, max_rounds: int = -1) -> None:
        """
        Run the game with a specific `GameState`
        :param initial_game_state: The initial `GameState` to use for running the game
        :param max_rounds: The number maximum of rounds to play the game
        """
        episode_id = 0
        print_every = int(max_rounds * PRINT_EACH)
        csv_log_every = int(max_rounds * CSV_LOG_EACH)
        agent_1_wins = 0.0
        agent_2_wins = 0.0
        mean_run_duration = 0.0
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
            agent_1_wins += stats['mean_accumulated_reward_sum_a1']
            agent_2_wins += stats['mean_accumulated_reward_sum_a2']
            mean_run_duration += stats['mean_action_duration_sum_a1'] + stats['mean_action_duration_sum_a2']
            if mr % csv_log_every == 0:
                self.csv_data['round_number'] = mr
                self.csv_data['agent1_mean_action_duration_sum'] = stats['mean_action_duration_sum_a1']
                self.csv_data['agent1_mean_accumulated_reward_sum'] = agent_1_wins
                self.csv_data['agent2_mean_action_duration_sum'] = stats['mean_action_duration_sum_a2']
                self.csv_data['agent2_mean_accumulated_reward_sum'] = agent_2_wins
                self.csv_data['mean_run_duration'] = mean_run_duration
                dw.writerow(self.csv_data)
            episode_id += 1

    @staticmethod
    def random_rollout_run(igs: GameState, rollouts: int) -> Tuple:
        """
        Random Rollout Runner to implement for a given GameRunner
        :param igs: Initial Game State to rollout on
        :param rollouts: The number of rollout to do
        :return: The tuple representing the history of games
        """
        raise NotImplementedError('Random Rollout not available for this GameRunner, or not implemented yet')

    def __del__(self) -> None:
        """
        Actions to do when the object is cleanup by the VM
        """
        self.csv_writer.close()
