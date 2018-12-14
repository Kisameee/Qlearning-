#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining GameStates
"""


from typing import Iterable

from .information_state import InformationState


class GameState:
    """
    Game State Base class
    WARNING : Action space must be finite for step function !
    """

    def step(self, player_id: int, action_id: int) -> ('GameState', float, bool):
        raise NotImplementedError

    def get_player_count(self) -> int:
        raise NotImplementedError

    def get_current_player_id(self) -> int:
        raise NotImplementedError

    def get_information_state_for_player(self, player_id: int) -> 'InformationState':
        raise NotImplementedError

    def get_available_actions_id_for_player(self, player_id: int) -> Iterable[int]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def copy_game_state(self) -> None:
        raise NotImplementedError

    def get_current_scores(self) -> None:
        raise NotImplementedError
