#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining Random Agent
"""


import random
from typing import Iterable

import numpy as np

from reinforcement_ecosystem.environments import InformationState, Agent, GameRunner


class RandomAgent(Agent):
    """
    Random Agent class for playing with it
    """

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `RandomAgent` does nothing
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        pass

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        action_count = len(available_actions)
        return available_actions[random.randint(0, action_count - 1)]


class RandomRolloutAgent(Agent):
    """
    Random Rollout Agent class for playing with it
    """

    def __init__(self, num_rollouts_per_available_action, runner: GameRunner):
        self.num_rollouts_per_available_action = num_rollouts_per_available_action
        self.runner = runner

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `RandomRolloutAgent` does nothing
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        pass

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `RandomRolloutAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        actions = tuple(available_actions)
        action_count = len(actions)
        action_scores = np.zeros(action_count)
        for i in range(action_count):
            gs = information_state.create_game_state_from_information_state()
            (result_gs, score, terminal) = gs.step(player_index, actions[i])

            # Two player zero sum game hypothesis
            player_score = (1 if player_index == 0 else -1) * score
            if not terminal:
                history = self.runner.run(self.num_rollouts_per_available_action, gs)
                player_score += history[player_index] - history[(player_index + 1) % 2]
            player_score = player_score / (1.0 if terminal else self.num_rollouts_per_available_action)
            action_scores[i] = player_score
        return actions[np.argmax(action_scores)]
