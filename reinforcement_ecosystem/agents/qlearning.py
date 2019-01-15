#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining QLearning Agent
"""


import random
from typing import Iterable

import numpy as np
import keras
from keras import Input, Model
from keras.activations import relu, linear
from keras.constraints import maxnorm
from keras.layers import Dense, Concatenate
from keras.optimizers import rmsprop

from reinforcement_ecosystem.environments import InformationState, Agent


class DeepQLearningAgent(Agent):
    """
    Deep QLearning Agent class for playing with it
    """

    def __init__(self, input_size: int, action_size: int, num_layers: int = 3, num_hidden_per_layer: int = 256,
                 epsilon: float = 0.01, lr: float = 0.01, gamma: float = 0.9, use_target: bool = True,
                 target_update_every: int = 100, print_error: bool = True, print_error_every: int = 100):
        """
        Initializer of the `DeepQLearningAgent`
        :param input_size: The input size of the model
        :param action_size: The action size to choose from
        :param num_layers: The number of layers of the model
        :param num_hidden_per_layer: The number of hidden layer per layer for the model
        :param epsilon: ???
        :param lr: The learning rate of the model
        :param gamma: ???
        :param use_target: ???
        :param target_update_every: ???
        :param print_error: If the class should print error
        :param print_error_every: To print error every X
        """
        self.lr = lr
        self.use_target = use_target
        self.target_update_every = target_update_every
        self.print_error = print_error
        self.print_error_every = print_error_every
        self.Q = self._create_net(input_size, action_size, num_layers, num_hidden_per_layer)
        self.QTarget = self._create_net(input_size, action_size, num_layers, num_hidden_per_layer)
        self.QTarget.set_weights(self.Q.get_weights())
        self.s = None
        self.a = None
        self.r = None
        self.t = None
        self.s_next_duplicated = None
        self.s_next_available_actions = None
        self.game_count = 0
        self.epsilon = epsilon
        self.gamma = gamma
        self.input_size = input_size
        self.action_size = action_size
        self.reward_history = np.array((0, 0, 0))
        self.accumulated_error = 0.0
        self.learn_steps = 0

    def _create_net(self, input_size: int, action_size: int, num_layers: int, num_hidden_per_layer: int) -> Model:
        """
        Build a neural network from the options passed as arguments
        :param input_size: The input size of the model
        :param action_size: The action size to choose from
        :param num_layers: The number of layers of the model
        :param num_hidden_per_layer: The number of hidden layer per layer for the model
        :return: A keras `Model`
        """
        input_state = Input(shape=(input_size,))
        input_action = Input(shape=(action_size,))
        inputs = Concatenate()([input_state, input_action])
        hidden = inputs
        for i in range(num_layers):
            hidden = Dense(num_hidden_per_layer, activation=relu, kernel_constraint=maxnorm(),
                           bias_constraint=maxnorm())(hidden)
        q = Dense(1, activation=linear)(hidden)
        model = Model([input_state, input_action], q)
        model.compile(optimizer=rmsprop(self.lr), loss=keras.losses.mse)
        return model

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `DeepQLearningAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if self.s is not None:
            self.r = (self.r if self.r else 0.0) + reward
            self.t = terminal
            if terminal:
                self.reward_history += (1 if reward == 1 else 0, 1 if reward == -1 else 0, 1 if reward == 0 else 0)
                self.learn()
                self.game_count += 1
                if (self.use_target and self.game_count % self.target_update_every) == 0:
                    if __debug__:
                        print('Updating Target Network')
                    self.QTarget.set_weights(self.Q.get_weights())

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `DeepQLearningAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        available_actions_list = list(available_actions)
        inputs_states = np.array([information_state.vectorize()] * len(available_actions_list))
        actions_vectorized = np.array(
            [keras.utils.to_categorical(action_id, self.action_size) for action_id in available_actions_list])
        if self.s is not None:
            self.s_next_duplicated = inputs_states
            self.s_next_available_actions = actions_vectorized
            self.t = False
            self.learn()
        if random.random() > self.epsilon:
            q_values = self.Q.predict([inputs_states, actions_vectorized]).flatten()
            best_id = q_values.argmax()
        else:
            best_id = random.randint(0, len(available_actions_list) - 1)
        self.s = inputs_states[best_id]
        self.a = actions_vectorized[best_id]
        return available_actions_list[best_id]

    def learn(self) -> None:
        """
        Make the Agent (model) learn
        """
        self.learn_steps += 1
        target = self.r + 0 if self.t else (
                self.gamma * (self.QTarget if self.use_target else self.Q)
                .predict([self.s_next_duplicated, self.s_next_available_actions])
                .flatten().max()
        )
        loss = self.Q.train_on_batch([np.array([self.s]), np.array([self.a])], np.array([target]))
        self.accumulated_error += loss
        self.s = None
        self.a = None
        self.r = None
        self.t = None
        self.s_next_duplicated = None
        self.s_next_available_actions = None
        if self.print_error and self.learn_steps % self.print_error_every == 0:
            if __debug__:
                print(self.accumulated_error / self.learn_steps)
            self.accumulated_error = 0


class TabularQLearningAgent(Agent):
    """
    Tabular QLearning Agent class for playing with it
    """

    def __init__(self):
        """
        Initializer of the `TabularQLearningAgent`
        """
        self.Q = dict()
        self.s = None
        self.a = None
        self.r = None
        self.t = None
        self.s_next = None
        self.game_count = 0
        self.reward_history = np.array((0, 0, 0))

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `TabularQLearningAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if self.s is not None:
            self.r = (self.r if self.r else 0) + reward
            self.t = terminal
            if terminal:
                self.reward_history += (1 if reward == 1 else 0, 1 if reward == -1 else 0, 1 if reward == 0 else 0)
                self.learn()
                self.s = None
                self.a = None
                self.r = None
                self.t = None
                self.game_count += 1
                if (self.game_count % 1000) == 0:
                    if __debug__:
                        print(self.reward_history / 1000)
                    self.reward_history = np.array((0, 0, 0))

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `TabularQLearningAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        if not (information_state in self.Q):
            self.Q[information_state] = dict()
            for action in available_actions:
                self.Q[information_state][action] = 1.1
        if self.s is not None:
            self.s_next = information_state
            self.learn()
        best_action = None
        best_action_score = 0
        for action in available_actions:
            if best_action is None or best_action_score < self.Q[information_state][action]:
                best_action = action
                best_action_score = self.Q[information_state][action]
        self.s = information_state
        self.a = best_action
        return best_action

    def learn(self):
        """
        Make the Agent (model) learn
        """
        self.Q[self.s][self.a] += 0.1 * (_reward_scaler(self.r) +
                                        (0 if self.t else (0.9 * max(self.Q[self.s_next].values()))) -
                                        self.Q[self.s][self.a])


def _reward_scaler(reward: int) -> float:
    """
    Scale a reward for a model
    :param reward: The reward to be scaled
    :return: The scaled reward
    """
    if reward == 1:
        return 1.0
    if reward == 0:
        return 0.0
    return -1.0
