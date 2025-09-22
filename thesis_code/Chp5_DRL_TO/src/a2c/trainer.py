"""
A2C Trainer for C-core optimization
Handles training loop, experience collection, and evaluation
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from .agent import A2CAgent
from ..femm_environment import CcoreFemmEnv


class A2CTrainer:
    """
    A2C Trainer class
    
    Manages the training process including experience collection,
    agent training, evaluation, and logging.
    """
    
    def __init__(self, config: Dict[str, Any], results_dir: Path):
        """
        Initialize A2C trainer
        
        Args:
            config: Configuration dictionary
            results_dir: Directory to save results
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config['logging']['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment and agent
        self.env = CcoreFemmEnv(
            max_steps=config['environment']['max_steps'],
            env_dim=config['environment']['env_dim'],
            femm_path=config['paths']['femm_path'],
            render_path=str(self.results_dir / 'renders'),
            log_level=config['logging']['log_level']
        )
        
        self.agent = A2CAgent(config)
        
        # Training parameters
        self.max_episodes = config['training']['max_episodes']
        self.max_updates = config['training']['max_updates']
        self.batch_size = config['training']['batch_size']
        self.play_interval = config['training']['play_interval']
        self.save_frequency = config['logging']['save_frequency']
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        self.evaluation_results = []
        
        self.logger.info("A2C Trainer initialized")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Dictionary containing training results and metrics
        """
        self.logger.info("Starting A2C training...")
        start_time = time.time()
        
        # Initialize environment
        obs = self.env.reset()
        obs_history = [obs]
        
        episode_count = 0
        update_count = 0
        
        # Training loop
        while episode_count < self.max_episodes and update_count < self.max_updates:
            
            # Collect experience batch
            batch_data = self._collect_batch(obs_history[-1])
            
            # Train agent
            if len(batch_data['observations']) > 0:
                training_metrics = self._train_agent(batch_data)
                self.training_metrics.append(training_metrics)
                update_count += 1
                
                self.logger.debug(f"Update {update_count}: {training_metrics}")
            
            # Update episode count
            if batch_data['episode_finished']:
                episode_count += 1
                
                # Log episode results
                episode_reward = sum(batch_data['rewards'])
                episode_length = len(batch_data['rewards'])
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                self.logger.info(
                    f"Episode {episode_count}: Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, Iron Count={batch_data.get('iron_count', 0)}"
                )
                
                # Reset environment for next episode
                obs = self.env.reset()
                obs_history = [obs]
            else:
                obs_history.append(batch_data['next_obs'])
            
            # Evaluation
            if update_count % self.play_interval == 0:
                eval_results = self._evaluate_agent()
                self.evaluation_results.append(eval_results)
                
                self.logger.info(
                    f"Evaluation at update {update_count}: "
                    f"Reward={eval_results['reward']:.2f}, "
                    f"Force={eval_results['net_force']:.2f}"
                )
            
            # Save model
            if episode_count % self.save_frequency == 0 and episode_count > 0:
                self._save_checkpoint(episode_count, update_count)
        
        # Final evaluation
        final_eval = self._evaluate_agent()
        self.evaluation_results.append(final_eval)
        
        # Save final results
        training_time = time.time() - start_time
        results = self._compile_results(training_time, episode_count, update_count)
        self._save_results(results)
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        return results
    
    def _collect_batch(self, initial_obs: np.ndarray) -> Dict[str, Any]:
        """
        Collect a batch of experience
        
        Args:
            initial_obs: Initial observation
            
        Returns:
            Dictionary containing batch data
        """
        observations = []
        actions = []
        rewards = []
        values = []
        dones = []
        
        obs = initial_obs
        episode_finished = False
        step_count = 0
        
        # Collect experience for batch_size steps or until episode ends
        while len(observations) < self.batch_size and not episode_finished:
            # Get action and value
            action, value = self.agent.get_action(obs)
            
            # Store current data
            observations.append(obs.copy())
            actions.append(action)
            values.append(value)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            rewards.append(reward)
            dones.append(done)
            
            obs = next_obs
            step_count += 1
            
            if done:
                episode_finished = True
                break
        
        # Get next value for advantage computation
        if not episode_finished:
            _, next_value = self.agent.get_action(obs)
        else:
            next_value = 0.0
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'dones': np.array(dones),
            'next_value': next_value,
            'next_obs': obs,
            'episode_finished': episode_finished,
            'iron_count': info.get('iron_count', 0) if len(observations) > 0 else 0
        }
    
    def _train_agent(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the agent on collected batch
        
        Args:
            batch_data: Batch of experience data
            
        Returns:
            Training metrics dictionary
        """
        # Compute returns and advantages
        returns, advantages = self.agent.compute_returns_and_advantages(
            batch_data['rewards'],
            batch_data['values'],
            batch_data['dones'],
            batch_data['next_value']
        )
        
        # Train agent
        metrics = self.agent.train_step(
            batch_data['observations'],
            batch_data['actions'],
            returns,
            advantages
        )
        
        # Add additional metrics
        metrics.update({
            'mean_reward': np.mean(batch_data['rewards']),
            'mean_value': np.mean(batch_data['values']),
            'mean_advantage': np.mean(advantages),
            'batch_size': len(batch_data['observations'])
        })
        
        return metrics
    
    def _evaluate_agent(self) -> Dict[str, Any]:
        """
        Evaluate the agent's performance
        
        Returns:
            Evaluation results dictionary
        """
        # Create evaluation environment
        eval_env = CcoreFemmEnv(
            max_steps=self.config['environment']['max_steps'],
            env_dim=self.config['environment']['env_dim'],
            femm_path=self.config['paths']['femm_path'],
            render_path=str(self.results_dir / 'eval_renders'),
            log_level='WARNING'  # Reduce logging during evaluation
        )
        
        obs = eval_env.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        
        # Run evaluation episode
        while not done:
            action, _ = self.agent.get_action(obs)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            step_count += 1
            
            if self.config['logging']['render_evaluation']:
                eval_env.render()
        
        eval_env.close()
        
        return {
            'reward': total_reward,
            'length': step_count,
            'net_force': info.get('net_force', 0.0),
            'iron_count': info.get('iron_count', 0)
        }
    
    def _save_checkpoint(self, episode: int, update: int):
        """Save model checkpoint"""
        checkpoint_dir = self.results_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'model_ep{episode}_up{update}.h5'
        self.agent.save_model(str(checkpoint_path))
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _compile_results(self, training_time: float, 
                        episodes: int, updates: int) -> Dict[str, Any]:
        """Compile training results"""
        return {
            'training_time': training_time,
            'total_episodes': episodes,
            'total_updates': updates,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_metrics': self.training_metrics,
            'evaluation_results': self.evaluation_results,
            'config': self.config,
            'final_performance': {
                'mean_reward_last_10': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards),
                'mean_length_last_10': np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths),
                'best_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
                'final_evaluation': self.evaluation_results[-1] if self.evaluation_results else {}
            }
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save training results to file"""
        results_file = self.results_dir / 'training_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def close(self):
        """Clean up resources"""
        self.env.close()
        self.logger.info("Trainer closed")
