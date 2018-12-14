#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining Agents
"""


from typing import Iterable

from .information_state import InformationState


class Agent:
    """
    Agent base class
    WARNING : Action space must be finite for act method !
    """

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        raise NotImplementedError

    def observe(self, reward: float, terminal: bool) -> None:
        raise NotImplementedError
