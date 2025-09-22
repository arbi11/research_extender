"""
Q-Learning Trainer

This module contains the Q-Learning trainer that coordinates training between
the agent and environment.
"""

import numpy as np
import logging
from pathlib import Path
import json
from tqdm import tqdm


class QLearningTrainer:
    """
    Q-Learning trainer for topology optimization.

    This class handles the training loop, logging, and result saving for
    Q-Learning-based topology optimization.
    """

    def __init__(self, environment, config):
        """
        Initialize the Q-Learning trainer.

        Args:
            environment: Environment instance
            config (dict): Configuration parameters
        """
        self.env = environment
        self.config = config

        # Training parameters
        self.episodes = config['q_learning']['episodes']
        self.max_steps = config['q_learning']['max_steps']

        # Create agent
        state_size = np.prod(self.env.get_state_shape())
        action_size = self.env.get_action_size()
        self.agent = self.env.get_agent_class()(state_size, action_size, config)

        # Training history
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_exploration': [],
            'best_reward': float('-inf'),
            'best_topology': None
        }

        self.logger = logging.getLogger(__name__)

    def train(self, results_dir):
        """
        Train the Q-Learning agent.

        Args:
            results_dir (Path): Directory to save results
        """
        self.logger.info("Starting Q-Learning training")
        self.logger.info(f"Episodes: {self.episodes}")
        self.logger.info(f"Max steps per episode: {self.max_steps}")

        # Create results directory
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        for episode in tqdm(range(self.episodes), desc="Training Episodes"):
            episode_reward, episode_length, exploration_count = self.run_episode()

            # Update training history
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(episode_length)
            self.history['episode_exploration'].append(exploration_count)

            # Check for best topology
            if episode_reward > self.history['best_reward']:
                self.history['best_reward'] = episode_reward
                self.history['best_topology'] = self.env.get_current_topology()

            # Decay epsilon
            self.agent.decay_epsilon()

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.history['episode_rewards'][-10:])
                self.logger.info(f"Episode {episode + 1}: "
                               f"Avg Reward: {avg_reward:.3f}, "
                               f"Epsilon: {self.agent.epsilon:.3f}")

            # Save checkpoint
            if (episode + 1) % 50 == 0:
                self.save_checkpoint(results_dir, episode + 1)

        # Save final results
        self.save_results(results_dir)

        self.logger.info("Q-Learning training completed")

    def run_episode(self):
        """
        Run a single training episode.

        Returns:
            tuple: (total_reward, episode_length, exploration_count)
        """
        # Reset environment
        state = self.env.reset()
        total_reward = 0
        exploration_count = 0

        # Episode loop
        for step in range(self.max_steps):
            # Choose action
            if np.random.random() < self.agent.epsilon:
                action = self.env.action_space.sample()
                exploration_count += 1
            else:
                action = self.agent.choose_action(state)

            # Take action
            next_state, reward, done, info = self.env.step(action)

            # Update agent
            self.agent.update_q_table(state, action, reward, next_state)

            # Update state and reward
            state = next_state
            total_reward += reward

            # Check if episode is done
            if done:
                break

        episode_length = step + 1
        return total_reward, episode_length, exploration_count

    def test_agent(self, results_dir, num_episodes=5):
        """
        Test the trained agent.

        Args:
            results_dir (Path): Directory to save test results
            num_episodes (int): Number of test episodes
        """
        self.logger.info(f"Testing agent for {num_episodes} episodes")

        test_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'final_topologies': []
        }

        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            # Run episode with exploitation only
            for step in range(self.max_steps):
                action = self.agent.get_best_action(state)
                state, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                if done:
                    break

            # Save results
            test_results['episode_rewards'].append(episode_reward)
            test_results['episode_lengths'].append(episode_length)
            test_results['final_topologies'].append(self.env.get_current_topology())

            self.logger.info(f"Test Episode {episode + 1}: "
                           f"Reward: {episode_reward:.3f}, "
                           f"Length: {episode_length}")

        # Save test results
        test_file = results_dir / "test_results.json"
        with open(test_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        return test_results

    def save_checkpoint(self, results_dir, episode):
        """
        Save training checkpoint.

        Args:
            results_dir (Path): Directory to save checkpoint
            episode (int): Current episode number
        """
        checkpoint_dir = results_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save agent
        agent_file = checkpoint_dir / f"agent_episode_{episode}.npz"
        self.agent.save_q_table(agent_file)

        # Save history
        history_file = checkpoint_dir / f"history_episode_{episode}.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def save_results(self, results_dir):
        """
        Save final training results.

        Args:
            results_dir (Path): Directory to save results
        """
        # Save training history
        history_file = results_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Save agent
        agent_file = results_dir / "final_agent.npz"
        self.agent.save_q_table(agent_file)

        # Save statistics
        stats = self.agent.get_statistics()
        stats_file = results_dir / "agent_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Save best topology
        if self.history['best_topology'] is not None:
            best_topology_file = results_dir / "best_topology.json"
            with open(best_topology_file, 'w') as f:
                json.dump({
                    'topology': self.history['best_topology'],
                    'reward': self.history['best_reward']
                }, f, indent=2)

    def get_training_summary(self):
        """
        Get training summary.

        Returns:
            dict: Training summary statistics
        """
        if not self.history['episode_rewards']:
            return {}

        rewards = np.array(self.history['episode_rewards'])
        lengths = np.array(self.history['episode_lengths'])

        return {
            'total_episodes': len(rewards),
            'best_reward': float(np.max(rewards)),
            'average_reward': float(np.mean(rewards)),
            'final_reward': float(rewards[-1]),
            'average_length': float(np.mean(lengths)),
            'total_steps': int(np.sum(lengths)),
            'convergence_episode': int(np.argmax(rewards)) + 1
        }
