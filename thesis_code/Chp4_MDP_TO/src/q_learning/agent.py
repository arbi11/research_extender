"""
Q-Learning Agent

This module contains the Q-Learning agent implementation for topology optimization.
"""

import numpy as np
import random
from collections import defaultdict
import logging


class QLearningAgent:
    """
    Q-Learning agent for topology optimization.

    This agent learns to optimize electromagnetic device topologies using
    Q-Learning with epsilon-greedy exploration strategy.
    """

    def __init__(self, state_size, action_size, config):
        """
        Initialize the Q-Learning agent.

        Args:
            state_size (int): Size of the state space
            action_size (int): Size of the action space
            config (dict): Configuration parameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Q-Learning parameters
        self.learning_rate = config['q_learning']['learning_rate']
        self.gamma = config['q_learning']['gamma']
        self.epsilon = config['q_learning']['epsilon_max']
        self.epsilon_min = config['q_learning']['epsilon_min']
        self.epsilon_decay = config['q_learning']['epsilon_decay']

        # Initialize Q-table as dictionary for sparse state space
        self.q_table = defaultdict(lambda: np.zeros(self.action_size))
        self.q_count = defaultdict(lambda: np.zeros(self.action_size))

        # State tracking
        self.states = []
        self.current_episode = 0

        self.logger = logging.getLogger(__name__)

    def normalize_q_values(self, q_values):
        """
        Normalize Q-values for exploration.

        Args:
            q_values (np.array): Q-values for current state

        Returns:
            np.array: Normalized probabilities
        """
        unique = np.unique(q_values)
        if unique.size == 1:
            return np.ones(self.action_size) / self.action_size
        else:
            q_min = q_values.min()
            q_max = q_values.max()
            if q_max > q_min:
                normalized = (q_values - q_min) / (q_max - q_min)
                normalized = normalized / normalized.sum()
                return normalized
            else:
                return np.ones(self.action_size) / self.action_size

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy.

        Args:
            state (tuple): Current state

        Returns:
            int: Chosen action
        """
        state_key = tuple(state)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: choose based on normalized Q-values
            q_values = self.q_table[state_key]
            probabilities = self.normalize_q_values(q_values)
            action = np.random.choice(self.action_size, p=probabilities)
        else:
            # Exploitation: choose best action
            q_values = self.q_table[state_key]
            action = np.argmax(q_values)

        return action

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-Learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Get current Q-value
        current_q = self.q_table[state_key][action]

        # Get maximum Q-value for next state
        next_max_q = np.max(self.q_table[next_state_key])

        # Q-Learning update rule
        target_q = reward + self.gamma * next_max_q
        td_error = target_q - current_q

        # Update Q-value with learning rate
        self.q_table[state_key][action] += self.learning_rate * td_error

        # Update visit count
        self.q_count[state_key][action] += 1

    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * (1 - self.epsilon_decay)
        )

    def get_best_action(self, state):
        """
        Get the best action for a given state (exploitation only).

        Args:
            state: Current state

        Returns:
            int: Best action
        """
        state_key = tuple(state)
        q_values = self.q_table[state_key]
        return np.argmax(q_values)

    def save_q_table(self, filepath):
        """
        Save Q-table to file.

        Args:
            filepath (str): Path to save file
        """
        np.savez(filepath,
                q_table=dict(self.q_table),
                q_count=dict(self.q_count),
                epsilon=self.epsilon)

    def load_q_table(self, filepath):
        """
        Load Q-table from file.

        Args:
            filepath (str): Path to load file
        """
        data = np.load(filepath, allow_pickle=True)
        self.q_table.update(data['q_table'].item())
        self.q_count.update(data['q_count'].item())
        self.epsilon = data['epsilon']

    def get_statistics(self):
        """
        Get agent statistics.

        Returns:
            dict: Dictionary containing agent statistics
        """
        total_states = len(self.q_table)
        total_actions = sum(len(actions) for actions in self.q_count.values())

        return {
            'total_states_visited': total_states,
            'total_actions_taken': total_actions,
            'current_epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma
        }
