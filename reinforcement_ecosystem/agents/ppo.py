#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Module for defining PPO Agent
"""


from typing import Any, Iterable

import numpy as np
import tensorflow as tf
from keras.constraints import max_norm

from reinforcement_ecosystem.environments import InformationState, Agent


__all__ = ['PPOWithMultipleTrajectoriesMultiOutputsAgent']


class PPOBrain:
    """
    ???
    """
    def __init__(self, state_size: int, num_layers: int, num_neuron_per_layer: int, action_size: int,
                 clip_epsilon: float = 0.2, c1: int = 1, c2: int = 0, session: tf.Session = None) -> 'PPOBrain':
        """
        Initializer for the PPO brain
        :param state_size: ???
        :param num_layers: The number of layer for the Model
        :param num_neuron_per_layer: The number of neuron per layer for the model
        :param action_size: ???
        :param clip_epsilon: ???
        :param c1: ???
        :param c2: ??? # No entropy for now
        :param session: The Tensorflow Session to use
        """
        self.state_size = state_size
        self.num_layers = num_layers
        self.num_neuron_per_layer = num_neuron_per_layer
        self.action_size = action_size
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.states_ph = tf.placeholder(shape=(None, state_size), dtype=tf.float32)
        self.advantages_ph = tf.placeholder(dtype=tf.float32)
        self.accumulated_reward_ph = tf.placeholder(dtype=tf.float32)
        self.old_policies_ph = tf.placeholder(dtype=tf.float32)
        self.policy_output, self.value_output, self.train_op = self.create_model()
        if session:
            self.session = session
        else:
            self.session = tf.get_default_session()
        if not self.session:
            self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def create_model(self) -> tuple:
        """
        Create Model from the internal state of the class
        :return: ???
        """
        hidden_policy = self.states_ph
        for i in range(self.num_layers):
            hidden_policy = tf.layers.batch_normalization(hidden_policy)
            hidden_policy = tf.layers.dense(hidden_policy, self.num_neuron_per_layer,
                                            activation=tf.keras.activations.relu,
                                            kernel_constraint=max_norm(16),
                                            bias_constraint=max_norm(16))
        policy_output = (tf.layers.dense(hidden_policy, self.action_size,
                                         activation=tf.keras.activations.softmax,
                                         kernel_constraint=max_norm(16),
                                         bias_constraint=max_norm(16))
                         + 0.000001)  # for numeric stability
        hidden_value = self.states_ph
        for i in range(self.num_layers):
            hidden_value = tf.layers.batch_normalization(hidden_value)
            hidden_value = tf.layers.dense(hidden_value, self.num_neuron_per_layer,
                                           activation=tf.keras.activations.relu,
                                           kernel_constraint=max_norm(16),
                                           bias_constraint=max_norm(16))

        value_output = tf.squeeze(tf.layers.dense(hidden_value, 1, activation=tf.keras.activations.linear,
                                                  kernel_constraint=max_norm(16),
                                                  bias_constraint=max_norm(16)), 1)
        advantages = self.advantages_ph
        r = policy_output / self.old_policies_ph
        # Lclip
        policy_loss = -tf.minimum(tf.multiply(r, advantages),
                                  tf.multiply(tf.clip_by_value(r, 1 - self.clip_epsilon,
                                                               1 + self.clip_epsilon),
                                              advantages))
        value_loss = tf.reduce_mean(tf.square(value_output - self.accumulated_reward_ph))
        entropy_loss = -policy_output * tf.log(policy_output)
        full_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
        train_op = tf.train.AdamOptimizer().minimize(full_loss)
        return policy_output, value_output, train_op

    def predict_policy(self, state: Any) -> Any:
        """
        ???
        :param state: ???
        :return: ???
        """
        return self.session.run(self.policy_output, feed_dict={
            self.states_ph: [state]
        })[0]

    def predict_policy_and_value(self, state: Any) -> Any:
        """
        ???
        :param state: ???
        :return: ???
        """
        pol, val = self.session.run([self.policy_output, self.value_output], feed_dict={
            self.states_ph: [state]
        })
        return pol[0], val[0]

    def predict_policies(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run(self.policy_output, feed_dict={
            self.states_ph: states,
        })

    def predict_values(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run(self.value_output, feed_dict={
            self.states_ph: states
        })

    def predict_policies_and_values(self, states: Any) -> Any:
        """
        ???
        :param states: ???
        :return: ???
        """
        return self.session.run([self.policy_output, self.value_output], feed_dict={
            self.states_ph: states
        })

    def train_network(self, state: Any, advantage: Any, accumulated_reward: Any, old_policy: Any) -> Any:
        """
        Train the PPO Brain model
        :param state: ???
        :param advantage: ???
        :param accumulated_reward: ???
        :param old_policy: ???
        :return: ???
        """
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: [state],
            self.advantages_ph: [advantage],
            self.accumulated_reward_ph: [accumulated_reward],
            self.old_policies_ph: [old_policy]
        })

    def train_network_batch(self, states: Any, advantages: Any, accumulated_rewards: Any, old_policies: Any) -> Any:
        """
        Train the PPO Brain model in batch
        :param states: ???
        :param advantages: ???
        :param accumulated_rewards: ???
        :param old_policies: ???
        :return: ???
        """
        return self.session.run(self.train_op, feed_dict={
            self.states_ph: states,
            self.advantages_ph: advantages,
            self.accumulated_reward_ph: accumulated_rewards,
            self.old_policies_ph: old_policies
        })


class PPOWithMultipleTrajectoriesMultiOutputsAgent(Agent):
    """
    PPOWithMultipleTrajectoriesMultiOutputsAgent class for playing with it
    """

    def __init__(self, state_size: int, action_size: int, num_layers: int = 5, num_neuron_per_layer: int = 64,
                 train_every_x_trajectories: int = 64, gamma: float = 0.9999,
                 num_epochs: int = 4, batch_size: int = 256):
        """
        Initializer for PPOWithMultipleTrajectoriesMultiOutputsAgent clas
        :param state_size: ???
        :param action_size: ???
        :param num_layers: The number of layer for the model
        :param num_neuron_per_layer: The number of neuron per layer for the model
        :param train_every_x_trajectories: ???
        :param gamma: ???
        :param num_epochs: The number of epochs to train the model on
        :param batch_size: The batch size for the the model to work with
        """
        self.brain = PPOBrain(state_size, num_layers, num_neuron_per_layer, action_size)
        self.action_size = action_size
        self.gamma = gamma
        self.trajectories = []
        self.current_trajectory_buffer = []
        self.train_every_x_trajectory = train_every_x_trajectories
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def act(self, player_index: int, information_state: InformationState, available_actions: Iterable[int]) -> int:
        """
        Play the given action for the `PPOWithMultipleTrajectoriesMultiOutputsAgent`
        :param player_index: The ID of the player playing
        :param information_state: The `InformationState` of the game
        :param available_actions: The legal action to choose from
        :return: The selected action
        """
        available_actions = list(available_actions)
        num_actions = len(available_actions)
        vectorized_state = information_state.vectorize()
        full_actions_probability, value = self.brain.predict_policy_and_value(vectorized_state)
        available_actions_probabilities = full_actions_probability[available_actions]
        sum_available_action_probabilities = np.sum(available_actions_probabilities)
        if sum_available_action_probabilities > 0.0000001:  # just in case all is zero, but unlikely
            probabilities = available_actions_probabilities / sum_available_action_probabilities
            chosen_index = np.random.choice(list(range(num_actions)), p=probabilities)
            chosen_action = available_actions[chosen_index]
        else:
            if __debug__:
                print("No action eligible, this should be extremely rare")
            chosen_index = np.random.choice(list(range(num_actions)))
            chosen_action = available_actions[chosen_index]
        transition = dict()
        transition['s'] = vectorized_state
        transition['a'] = chosen_action
        transition['r'] = 0.0
        transition['t'] = False
        transition['p_old'] = full_actions_probability.tolist()
        self.current_trajectory_buffer.append(transition)
        return chosen_action

    def observe(self, reward: float, terminal: bool) -> None:
        """
        Observe the state of the game for the `PPOWithMultipleTrajectoriesMultiOutputsAgent`
        :param reward: Reward of the player after the game
        :param terminal: If the game is in a terminal mode
        """
        if not self.current_trajectory_buffer:
            return
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['r'] += reward
        self.current_trajectory_buffer[len(self.current_trajectory_buffer) - 1]['t'] |= terminal
        if terminal:
            accumulated_reward = 0.0
            for t in reversed(range(len(self.current_trajectory_buffer))):
                accumulated_reward = self.current_trajectory_buffer[t]['r'] + self.gamma * accumulated_reward
                self.current_trajectory_buffer[t]['R'] = accumulated_reward
            self.trajectories.append(self.current_trajectory_buffer)
            self.current_trajectory_buffer = []
            if len(self.trajectories) == self.train_every_x_trajectory:
                transitions = [transition for trajectory in self.trajectories for transition in trajectory]
                states = np.array(
                    [transition['s'] for transition in transitions])
                accumulated_rewards = np.array(
                    [transition['R'] for transition in transitions])
                old_policies = np.array(
                    [transition['p_old'] for transition in transitions])
                num_samples = states.shape[0]
                batch_size = min(self.batch_size, num_samples)
                indexes = np.array(list(range(num_samples)))
                for i in range(self.num_epochs):
                    np.random.shuffle(indexes)
                    index_batch = indexes[0:batch_size]
                    states_batch = states[index_batch]
                    accumulated_batch = accumulated_rewards[index_batch]
                    advantages_batch = np.zeros((batch_size, self.action_size))
                    single_dimension_advantages = accumulated_batch - self.brain.predict_values(states_batch)
                    for idx in range(batch_size):
                        advantages_batch[idx, transitions[index_batch[idx]]['a']] = single_dimension_advantages[idx]
                    old_policies_batch = old_policies[index_batch]
                    self.brain.train_network_batch(states_batch, advantages_batch, accumulated_batch, old_policies_batch)
                self.trajectories = []
