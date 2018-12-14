#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining MOIMSCTS Agent
"""


import random
from math import sqrt, log
from typing import Any, Iterable, Tuple

import numpy as np
from keras.activations import softmax
from keras.activations import relu, linear
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.layers.core import dense
import tensorflow as tf

from reinforcement_ecosystem.environments import InformationState, GameRunner, GameState, Agent


__all__ = ['MOISMCTSWithRandomRolloutsAgent', 'MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent',
           'MOISMCTSWithValueNetworkAgent']


class ValueNetworkBrain:
    """
    ValueNetworkBrain class for ???
    """

    def __init__(self, state_size, num_players: int, num_layers: int = 2, num_neurons_per_layer: int = 512,
                 session: tf.Session = None) -> None:
        """
        Initializer for the `ValueNetworkBrain` class
        :param state_size: ???
        :param num_players: ???
        :param num_layers: The number of layer for the model
        :param num_neurons_per_layer: The number of neurons per layers
        :param session: The `tf.Session` to use
        """
        self.state_size = state_size
        self.num_players = num_players
        self.num_layers = num_layers
        self.num_neurons_per_layers = num_neurons_per_layer
        self.states_ph = tf.placeholder(shape=(None, state_size), dtype=tf.float64)
        self.target_values_ph = tf.placeholder(shape=(None, num_players), dtype=tf.float64)
        self.values_op, self.train_op = self.create_network()
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        if not self.session:
            self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_network(self) -> Any:
        """
        Build a neural network from the options passed as arguments
        :return: ???
        """
        hidden = self.states_ph
        for i in range(self.num_layers):
            hidden = dense(hidden, self.num_neurons_per_layers, activation=relu)
        values_op = dense(hidden, self.num_players, activation=linear)
        loss = tf.reduce_mean(tf.square(values_op - self.target_values_ph))
        train_op = tf.train.AdamOptimizer().minimize(loss)
        return values_op, train_op

    def predict_state(self, state: Any) -> Any:
        """
        ???
        :param state: ???
        :return: ???
        """
        return self.predict_states([state])[0]

    def predict_states(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run(self.values_op, feed_dict={self.states_ph: states})

    def train(self, states: Any, target_values: Any) -> Any:
        """
        ???
        :param states: ???
        :param target_values: ???
        :return: ???
        """
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: states,
            self.target_values_ph: target_values
        })


class MOIMCTSMixin:
    """
    MOISCMCTS Mixin for
    """

    def __init__(self, k: float = 0.2) -> None:
        """
        Initializer for the Mixin
        """
        self.current_iteration_selected_nodes = {}
        self.current_trees = {}
        self.k = k

    def add_visited_node(self, node: dict, selected_action: int, current_player: int) -> None:
        """
        Add the visited node to the current iteration  for a given player and action ???
        :param node: The node to be added
        :param selected_action: The selected action
        :param current_player: The current player ID
        """
        if current_player not in self.current_iteration_selected_nodes:
            self.current_iteration_selected_nodes[current_player] = []
        self.current_iteration_selected_nodes[current_player].append((node, selected_action))

    def select(self, gs: GameState) -> Tuple:
        """
        ???
        :param gs: The Game state to select on ???
        """
        terminal = False
        while True:
            current_player = gs.get_current_player_id()
            info_state = gs.get_information_state_for_player(current_player)
            if terminal:
                return gs, info_state, current_player, True
            if current_player not in self.current_trees:
                self.current_trees[current_player] = {}
            current_tree = self.current_trees[current_player]
            if info_state not in current_tree:
                current_tree[info_state] = {'nprime': 0}
                return gs, info_state, current_player, False
            current_node = current_tree[info_state]
            child_action = max(current_node['a'],
                               key=lambda node:
                               ((node['r'] / node['n'] + self.k * sqrt(log(current_node['nprime']) / node['n']))
                                   if node['n'] > 0 else 99999999))
            action_to_execute = child_action['action_id']
            self.add_visited_node(current_node, child_action, current_player)
            gs, reward, terminal = gs.step(current_player, action_to_execute)


class MOISMCTSWithRandomRolloutsAgent(Agent, MOIMCTSMixin):
    """
    Deep MOISMCTS with random rollouts Agent class for playing with it
    """

    def __init__(self, iteration_count: int, runner: GameRunner, reuse_tree: bool = True, k: float = 0.2):
        """
        Initializer for the `MOISMCTSWithRandomRolloutsAgent` class
        :param iteration_count: ???
        :param runner: ???
        :param reuse_tree: If we should reuse an existing tree
        :param k: ???
        """
        super(MOIMCTSMixin, self).__init__(k)
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.runner = runner

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent` does nothing
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
        for i in range(self.iteration_count):
            self.current_iteration_selected_nodes = {}
            gs = information_state.create_game_state_from_information_state()
            # SELECT
            gs, info_state, current_player, terminal = self.select(gs)
            if not terminal:
                # EXPAND
                node = self.current_trees[current_player][info_state]
                available_actions = gs.get_available_actions_id_for_player(current_player)
                node['a'] = [{'n': 0, 'r': 0, 'action_id': action_id} for action_id in available_actions]
                child_action = random.choice(node['a'])
                action_to_execute = child_action['action_id']
                self.add_visited_node(node, child_action, current_player)
                gs, reward, terminal = gs.step(current_player, action_to_execute)
            # EVALUATE
            scores = self.runner.run(initial_game_state=gs, max_rounds=1)
            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]
        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'],
                           key=lambda child: child['n'])
        return child_action['action_id']


class MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent(Agent, MOIMCTSMixin):
    """
    MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent class for playing with it
    """

    def __init__(self, iteration_count: int, runner: GameRunner, state_size, action_size, reuse_tree=True,
                 training_episodes=3000, evaluation_episodes=1000, k=0.2) -> None:
        """
        Initializer for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent` class
        :param iteration_count: ???
        :param runner: ???
        :param state_size: ???
        :param action_size: ???
        :param reuse_tree: If we should reuse an existing tree or not
        :param training_episodes: ???
        :param evaluation_episodes: ???
        :param k: ???
        """
        super(MOIMCTSMixin, self).__init__(k)
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.training_episodes = training_episodes
        self.evaluation_episodes = evaluation_episodes
        self.state_size = state_size
        self.action_size = action_size
        self.runner = runner
        self.current_episode = 0
        self.X = []
        self.Y = []
        self.model = self.create_model()

    def create_model(self) -> 'keras.models.Model':
        """
        Build frol ????
        :return: A build keras model
        """
        model = Sequential()
        model.add(Dense(512, activation=relu, input_dim=self.state_size))
        model.add(Dense(512, activation=relu))
        model.add(Dense(self.action_size, activation=softmax))
        model.compile(optimizer=Adam(), loss=categorical_crossentropy)
        return model

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if terminal:
            self.current_episode += 1
            if self.current_episode == self.training_episodes:
                self.model.fit(np.array(self.X), np.array(self.Y), 512, epochs=2048)

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `MOISMCTSWithRandomRolloutsExpertThenApprenticeAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        available_actions = list(available_actions)
        if self.evaluation_episodes + self.training_episodes < self.current_episode:
            self.X = []
            self.Y = []
            self.current_episode = 0
        if self.current_episode > self.training_episodes:
            probs = self.model.predict(np.array([information_state.vectorize()]))[0]
            available_probs = probs[np.array(available_actions)]
            probs_sum = np.sum(available_probs)
            if probs_sum > 0.001:
                chosen_action_index = np.argmax(available_probs)
                action = available_actions[chosen_action_index]
            else:
                action = random.choice(available_actions)
            return action
        for i in range(self.iteration_count):
            self.current_iteration_selected_nodes = {}
            gs = information_state.create_game_state_from_information_state()
            # SELECT
            gs, info_state, current_player, terminal = self.select(gs)
            if not terminal:
                # EXPAND
                node = self.current_trees[current_player][info_state]
                available_actions = gs.get_available_actions_id_for_player(current_player)
                node['a'] = [{'n': 0, 'r': 0, 'action_id': action_id} for action_id in available_actions]
                child_action = random.choice(node['a'])
                action_to_execute = child_action['action_id']
                self.add_visited_node(node, child_action, current_player)
                gs, reward, terminal = gs.step(current_player, action_to_execute)
            # EVALUATE
            scores = self.runner.run(initial_game_state=gs, max_rounds=1)
            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]
        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'], key=lambda child: child['n'])
        self.X.append(information_state.vectorize().tolist())
        self.Y.append(to_categorical(child_action['action_id'], self.action_size))
        return child_action['action_id']


class MOISMCTSWithValueNetworkAgent(Agent, MOIMCTSMixin):
    """
    MOISMCTSWithValueNetworkAgent class for playing with it
    """

    def __init__(self, iteration_count: int, state_size: int, num_players: int,
                 brain: ValueNetworkBrain = None, reuse_tree: bool = True, k: float = 0.2, gamma: float = 0.99):
        """
        Initializer for `MOISMCTSWithValueNetworkAgent`
        :param iteration_count: ???
        :param state_size: ???
        :param num_players: The number of players
        :param brain: ???
        :param reuse_tree: If we should reuse the tree or not
        :param k: ???
        :param gamma: ???
        """
        super(MOIMCTSMixin, self).__init__(k)
        self.iteration_count = iteration_count
        self.reuse_tree = reuse_tree
        self.gamma = gamma
        self.brain = brain
        self.num_players = num_players
        self.state_size = state_size
        if not brain:
            self.brain = ValueNetworkBrain(state_size, num_players)
        self.current_trajectory = []
        self.current_transition = None

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `MOISMCTSWithValueNetworkAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.current_transition:
            return
        self.current_transition['r'] += reward
        self.current_transition['terminal'] |= terminal
        if terminal:
            R = 0
            self.current_trajectory.append(self.current_transition)
            self.current_transition = None
            for transition in reversed(self.current_trajectory):
                R = transition['r'] + self.gamma * R
                accumulated_rewards = np.ones(self.num_players) * R
                for i in range(self.num_players):
                    if i != transition['player_index']:
                        accumulated_rewards[i] = -R / (self.num_players - 1)
                transition['R'] = accumulated_rewards
            states = np.array([transition['s'] for transition in self.current_trajectory])
            target_values = np.array([transition['R'] for transition in self.current_trajectory])
            self.brain.train(states, target_values)
            self.current_trajectory = []

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `MOISMCTSWithValueNetworkAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        if self.current_transition:
            self.current_transition['terminal'] = False
            self.current_trajectory.append(self.current_transition)
            self.current_transition = None
        for i in range(self.iteration_count):
            self.current_iteration_selected_nodes = {}
            gs = information_state.create_game_state_from_information_state()
            # SELECT
            gs, info_state, current_player, terminal = self.select(gs)
            if not terminal:
                # EXPAND
                node = self.current_trees[current_player][info_state]
                available_actions = gs.get_available_actions_id_for_player(current_player)
                node['a'] = [{'n': 0, 'r': 0, 'action_id': action_id} for action_id in available_actions]
                child_action = random.choice(node['a'])
                action_to_execute = child_action['action_id']
                self.add_visited_node(node, child_action, current_player)
                gs, reward, terminal = gs.step(current_player, action_to_execute)
            # EVALUATE
            scores = self.brain.predict_state(info_state.vectorize())
            # BACKPROPAGATE SCORE
            for player_id in self.current_iteration_selected_nodes.keys():
                visited_nodes = self.current_iteration_selected_nodes[player_id]
                for node, child_action in reversed(visited_nodes):
                    node['nprime'] += 1
                    child_action['n'] += 1
                    child_action['r'] += scores[player_id]
        child_action = max(self.current_iteration_selected_nodes[player_index][0][0]['a'], key=lambda child: child['n'])
        self.current_transition = {
            's': information_state.vectorize(),
            'r': 0,
            'player_index': player_index,
            'terminal': False
        }
        return child_action['action_id']
