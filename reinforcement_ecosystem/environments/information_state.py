#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining Information States
"""


from typing import Hashable


class InformationState:

    def __hash__(self) -> Hashable:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def vectorize(self):
        raise NotImplementedError

    def create_game_state_from_information_state(self):
        raise NotImplementedError
