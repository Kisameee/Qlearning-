#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for for playing tic tac toe
"""


from time import time
from typing import Any, Hashable, Iterable

import numpy as np

from reinforcement_ecosystem.environments import InformationState, GameState, GameRunner, Agent
from reinforcement_ecosystem.agents import RandomAgent
from reinforcement_ecosystem.config import TF_LOG_DIR


class TicTacToeRunner(GameRunner):
    """
    Tic Tac Toe Game Runner which run a Game of Tic Tac Toe between two players
    """

    def __init__(self, agent1: Agent, agent2: Agent, tf_log_dir: str) -> None:
        """
        Initializer for Game Runner
        :param agent1: The first player agent
        :param agent2: The second player agent
        :param tf_log_dir: Where Tensorflow should log
        """
        super(TicTacToeRunner, self).__init__(agent1, agent2, tf_log_dir)

    def _run(self, initial_game_state: GameState) -> dict:
        """
        Run a game of TicTacToe for a given GameState
        :param initial_game_state: The initial GameState to play with
        :return: A dict of stats for the game
        """
        gs = initial_game_state.copy_game_state()
        scores, terminal = gs.get_current_scores()
        round_step = 0
        mean_action_duration_sum = {0: 0.0, 1: 0.0}
        mean_accumulated_reward_sum = {0: 0.0, 1: 0.0}
        while not terminal:
            current_player = gs.get_current_player_id()
            action_ids = gs.get_available_actions_id_for_player(current_player)
            info_state = gs.get_information_state_for_player(current_player)
            action_time = time()
            action = self.agents[current_player].act(current_player, info_state, action_ids)
            action_time = time() - action_time
            # WARNING : Two Players Zero Sum Game Hypothesis
            (gs, score, terminal) = gs.step(current_player, action)
            self.agents[current_player].observe((1 if current_player == 0 else -1) * score, terminal)
            mean_accumulated_reward_sum[current_player] = score
            mean_action_duration_sum[current_player] += action_time
            round_step += 1
        stats = {
            'round_step': round_step,
            'mean_action_duration_sum_a1': mean_action_duration_sum[0],
            'mean_action_duration_sum_a2': mean_action_duration_sum[1],
            'mean_accumulated_reward_sum_a1': mean_accumulated_reward_sum[0],
            'mean_accumulated_reward_sum_a2': mean_accumulated_reward_sum[1]
        }
        return stats


class TicTacToeInformationState(InformationState):
    """
    TicTacToe Information state
    """

    def __hash__(self) -> Hashable:
        """
        Conpute the Hash of an TicTacToeInformationState
        :return: Hash of an TicTacToeInformationState
        """
        return hash((self.board * (1 if self.current_player == 0 else -1)).tobytes())
        # sum = self.current_player
        # for i in range(9):
        #    sum += 3 ^ (i + 1) + (self.board[i // 3, i % 3] + 1)
        # return int(sum)

    def __eq__(self, other: 'TicTacToeInformationState') -> bool:
        """
        Check equality between two TicTacToeInformationState
        :param other: The other TicTacToeInformationState to check equality with
        :return: If the two TicTacToeInformationState are equals
        """
        # if isinstance(other, TicTacToeInformationState):
        #    return False
        return np.array_equal(self.board, other.board) and self.current_player == other.current_player

    def __ne__(self, other: 'TicTacToeInformationState') -> bool:
        """
        Check non equality between two TicTacToeInformationState
        :param other: The other TicTacToeInformationState to check non equality with
        :return: If the two TicTacToeInformationState are non equals
        """
        # if isinstance(other, TicTacToeInformationState):
        #    return False
        return not (np.array_equal(self.board, other.board) and self.current_player == other.current_player)

    def __init__(self, current_player: int, board: Any):
        """
        Initializer for TicTacToeInformationState
        :param current_player: ???
        :param board: ???
        """
        self.current_player = current_player
        self.board = board

    def __str__(self):
        """
        Compute the InformationState as str
        :return: The InformationState as str
        """
        str_acc = 'current player : ' + str(self.current_player) + '\n'
        for i in range(0, 3):
            for j in range(0, 3):
                val = self.board[i][j]
                str_acc += '_' if val == 0 else ('0' if val == 1 else 'X')
            str_acc += '\n'
        return str_acc

    def vectorize(self) -> np.ndarray:
        """
        Vectorize the board
        :return: A vectorized TicTacToe InformationState
        """
        return self.board.reshape((9,))

    def create_game_state_from_information_state(self) -> GameState:
        """
        Create a new GameState with the current InformationState
        :return: A fresh GameState
        """
        gs = TicTacToeGameState()
        gs.board = self.board.copy()
        gs.current_player = self.current_player
        return gs


class TicTacToeGameState(GameState):
    """
    TicTacToe Game Stats which manage the game
    """

    def __init__(self):
        self.current_player = 0
        self.board = np.array(
            (
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0)
             )
        )

    def step(self, player_id: int, action_id: int) -> (GameState, float, bool):
        """
        Play a move for a given Player
        :param player_id: The player id to play with
        :param action_id: The action to be executed by the Player
        :return: The GameState with the score and the terminal state
        """
        if self.current_player != player_id:
            raise Exception('This is not this player turn !')
        val = self.board[action_id // 3][action_id % 3]
        if val != 0:
            raise Exception("Player can't play at specified position !")
        self.board[action_id // 3][action_id % 3] = \
            1 if player_id == 0 else -1
        (score, terminal) = self.compute_current_score_and_end_game_more_efficient()
        self.current_player = (self.current_player + 1) % 2
        return self, score, terminal

    # def compute_current_score_and_end_game(self)-> (float, bool):
    def compute_current_score_and_end_game_more_efficient(self) -> (float, bool):
        """
        Check end game
        :return: ???
        """
        if self.board[0][0] + self.board[0][1] + self.board[0][2] == 3 or \
                self.board[1][0] + self.board[1][1] + self.board[1][2] == 3 or \
                self.board[2][0] + self.board[2][1] + self.board[2][2] == 3 or \
                self.board[0][0] + self.board[1][0] + self.board[2][0] == 3 or \
                self.board[0][1] + self.board[1][1] + self.board[2][1] == 3 or \
                self.board[0][2] + self.board[1][2] + self.board[2][2] == 3 or \
                self.board[0][0] + self.board[1][1] + self.board[2][2] == 3 or \
                self.board[2][0] + self.board[1][1] + self.board[0][2] == 3:
            return 1.0, True
        if self.board[0][0] + self.board[0][1] + self.board[0][2] == -3 or \
                self.board[1][0] + self.board[1][1] + self.board[1][2] == -3 or \
                self.board[2][0] + self.board[2][1] + self.board[2][2] == -3 or \
                self.board[0][0] + self.board[1][0] + self.board[2][0] == -3 or \
                self.board[0][1] + self.board[1][1] + self.board[2][1] == -3 or \
                self.board[0][2] + self.board[1][2] + self.board[2][2] == -3 or \
                self.board[0][0] + self.board[1][1] + self.board[2][2] == -3 or \
                self.board[2][0] + self.board[1][1] + self.board[0][2] == -3:
            return -1.0, True
        if 0 in self.board:
            return 0.0, False
        return 0.0, True

    def get_player_count(self) -> int:
        """
        Get the player count for the game
        :return: The player count
        """
        return 2

    def get_current_player_id(self) -> int:
        """
        Get the id of the current player
        :return: ID of the player playing
        """
        return self.current_player

    def get_information_state_for_player(self, player_id: int) -> InformationState:
        """
        Get the legal information for a given player
        :param player_id: The player ID to check information
        :return: The available information for the player
        """
        return TicTacToeInformationState(self.current_player, self.board.copy())

    def get_available_actions_id_for_player(self, player_id: int) -> Iterable[int]:
        """
        Get the legal action for a given player
        :param player_id: The player ID to check moves
        :return: The available actions of the player
        """
        if player_id != self.current_player:
            return []
        return list(filter(lambda i: self.board[i // 3][i % 3] == 0, range(0, 9)))

    def __str__(self) -> str:
        """
        Compute the gamestate as str
        :return: The gamestate as str
        """
        s = ''
        for i in range(0, 3):
            for j in range(0, 3):
                val = self.board[i][j]
                s += '_' if val == 0 else ('0' if val == 1 else 'X')
            s += '\n'
        return s

    def copy_game_state(self) -> InformationState:
        """
        Copy the TicTacToe GS
        :return: A copied TicTacToe GS
        """
        gs = TicTacToeGameState()
        gs.board = self.board.copy()
        gs.current_player = self.current_player
        return gs

    def get_current_scores(self) -> tuple:
        """
        Get the score and the terminal state
        :return: The score as numpy array and the terminal state
        """
        winner, terminal = self.compute_current_score_and_end_game_more_efficient()
        return np.array([winner, -winner]), terminal


if __name__ == "__main__":
    print('Random vs Random')
    TicTacToeRunner(RandomAgent(), RandomAgent(), tf_log_dir=TF_LOG_DIR + '/Rdm_Vs_Rdm')\
        .run(TicTacToeGameState(), 1000)
