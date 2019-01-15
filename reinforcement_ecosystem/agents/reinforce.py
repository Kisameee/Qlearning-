#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining Reinforce Agent
"""


from typing import Any, Iterable

import numpy as np
from keras import Input, Model
from keras.activations import tanh, sigmoid
from keras.layers import Dense, concatenate
from keras.optimizers import adam
from keras.utils import to_categorical
import keras.backend as K

from reinforcement_ecosystem.environments import InformationState, Agent


class ReinforceClassicBrain:
    """
    Reinforce Classic Brain for ???
    """

    def __init__(self, state_size: int, num_layers: int, num_neuron_per_layer: int, action_size: int):
        """
        Initializer for `ReinforceClassicBrain`
        :param state_size: ???
        :param num_layers: The number of layer of the model
        :param num_neuron_per_layer: The number of neuron per layer
        :param action_size: The maximum number of action
        """
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_neuron_per_layer = num_neuron_per_layer
        self.action_size = action_size
        self.model = self.create_model()

    def create_model(self) -> Model:
        """
        Create a Model from the internal state of the class
        :return: A Keras `Model`
        """
        input_state = Input(shape=(self.state_size,))
        input_action = Input(shape=(self.action_size,))
        hidden = concatenate([input_state, input_action])
        for i in range(self.num_layers):
            hidden = Dense(self.num_neuron_per_layer, activation=tanh)(hidden)
        policy = Dense(1, activation=sigmoid)(hidden)
        model = Model([input_state, input_action], policy)
        model.compile(loss=ReinforceClassicBrain.reinforce_loss, optimizer=adam())
        return model

    def predict_policy(self, state: Any, action: Any) -> Any:
        """
        ???
        :param state: ???
        :param action: ???
        :return: ???
        """
        return self.model.predict([np.array([state]), np.array([action])])[0]

    def predict_policies(self, states: Any, actions: Any) -> Any:
        """
        ???
        :param states: ???
        :param actions: ???
        :return: ???
        """
        return self.model.predict([states, actions])

    @staticmethod
    def reinforce_loss(y_true: 'tensorflow.Tensor', y_pred: 'tensorflow.Tensor') -> 'tensorflow.Tensor':
        """
        Custom loss function for the reinforce model
        Compute the mean of the inverted log of the prediction by the product of the true output
        :param y_true: Real outputs
        :param y_pred: Predicted output of the model
        :return: The computed loss as `tensorflow.Tensor`
        """
        return K.mean(-K.log(y_pred) * y_true)

    def train_policy(self, state: Any, action: Any, advantage: Any) -> None:
        """
        ???
        :param state: ???
        :param action: ???
        :param advantage: ???
        :return: ???
        """
        self.model.train_on_batch([np.array([state]), np.array([action])], np.array([advantage]))

    def train_policies(self, states: Any, actions: Any, advantages: Any) -> None:
        """
        ???
        :param states: ???
        :param actions: ???
        :param advantages: ???
        """
        self.model.train_on_batch([states, actions], advantages)


class ReinforceClassicAgent(Agent):
    """
    Reinforce Classic Agent class for playing with it
    """

    def __init__(self, state_size: int, action_size: int, num_layers: int = 5, num_neuron_per_layer: int = 128) -> None:
        """
        Initializer for the `ReinforceClassicAgent` class
        :param state_size: ???
        :param action_size: ???
        :param num_layers: The number of layer for the Model
        :param num_neuron_per_layer: The number of neurons per layer for the model
        """
        self.brain = ReinforceClassicBrain(state_size, num_layers, num_neuron_per_layer, action_size)
        self.action_size = action_size
        self.episode_buffer = []

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `ReinforceClassicAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        vectorized_states = np.array([information_state.vectorize()] * len(available_actions))
        actions_vectorized = np.array([to_categorical(action, self.action_size) for action in available_actions])
        logits = self.brain.predict_policies(vectorized_states, actions_vectorized)
        sum = np.sum(logits)
        probabilities = np.reshape(logits / sum, (len(available_actions),))
        chosen_action = np.random.choice(available_actions, p=probabilities)
        transition = dict()
        transition['s'] = information_state.vectorize()
        transition['a'] = to_categorical(chosen_action, self.action_size)
        transition['r'] = 0.0
        transition['t'] = False
        self.episode_buffer.append(transition)
        return chosen_action

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `ReinforceClassicAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.episode_buffer:
            return
        self.episode_buffer[len(self.episode_buffer) - 1]['r'] += reward
        self.episode_buffer[len(self.episode_buffer) - 1]['t'] |= terminal
        if terminal:
            states = np.array([transition['s'] for transition in self.episode_buffer])
            actions = np.array([transition['a'] for transition in self.episode_buffer])
            R = 0.0
            for t in reversed(range(len(self.episode_buffer))):
                R = self.episode_buffer[t]['r'] + 0.9 * R
                self.episode_buffer[t]['R'] = R
            advantages = np.array([transition['R'] for transition in self.episode_buffer])
            self.brain.train_policies(states, actions, advantages)
            self.episode_buffer = []


class ReinforceClassicWithMultipleTrajectoriesAgent(Agent):
    """
    Reinforce Classic With Multiple Trajectories  Agent class for playing with it
    """

    def __init__(self, state_size: int, action_size: int, num_layers: int = 5,
                 num_neuron_per_layer: int = 128, train_every_X_trajectories: int = 16):
        """
        Initializer for the `ReinforceClassicWithMultipleTrajectoriesAgent` class
        :param state_size: ???
        :param action_size: ???
        :param num_layers: The number of layer for the Model
        :param num_neuron_per_layer: The number of neuron per layer for the model
        :param train_every_X_trajectories: ???
        """
        self.brain = ReinforceClassicBrain(state_size, num_layers, num_neuron_per_layer, action_size)
        self.action_size = action_size
        self.trajectories = []
        self.current_trajectory_buffer = []
        self.train_every_X_trajectory = train_every_X_trajectories

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `ReinforceClassicWithMultipleTrajectoriesAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        vectorized_states = np.array([information_state.vectorize()] * len(available_actions))
        actions_vectorized = np.array([to_categorical(action, self.action_size) for action in available_actions])
        logits = self.brain.predict_policies(vectorized_states, actions_vectorized)
        sum = np.sum(logits)
        probabilities = np.reshape(logits / sum, (len(available_actions),))
        chosen_action = np.random.choice(available_actions, p=probabilities)
        transition = dict()
        transition['s'] = information_state.vectorize()
        transition['a'] = to_categorical(chosen_action, self.action_size)
        transition['r'] = 0.0
        transition['t'] = False
        self.current_trajectory_buffer.append(transition)
        return chosen_action

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `ReinforceClassicWithMultipleTrajectoriesAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.current_trajectory_buffer:
            return
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['r'] += reward
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['t'] |= terminal
        if terminal:
            R = 0.0
            for t in reversed(range(len(self.current_trajectory_buffer))):
                R = self.current_trajectory_buffer[t]['r'] + 0.9 * R
                self.current_trajectory_buffer[t]['R'] = R
            self.trajectories.append(self.current_trajectory_buffer)
            self.current_trajectory_buffer = []
            if len(self.trajectories) == self.train_every_X_trajectory:
                states = np.array([transition['s'] for trajectory in self.trajectories for transition in trajectory])
                actions = np.array([transition['a'] for trajectory in self.trajectories for transition in trajectory])
                advantages = np.array([transition['R'] for trajectory in self.trajectories for transition in trajectory])
                self.brain.train_policies(states, actions, advantages)
                self.trajectories = []
