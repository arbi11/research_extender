"""
A2C (Advantage Actor-Critic) Agent for C-core optimization
Based on the original A2C implementation with TensorFlow 2.x
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
from typing import Tuple, Dict, Any
import logging


class ProbabilityDistribution(tf.keras.Model):
    """Probability distribution for action sampling"""
    
    def call(self, logits):
        """Sample a random categorical action from given logits"""
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class A2CModel(tf.keras.Model):
    """
    A2C neural network model with CNN feature extraction
    and separate actor-critic heads
    """
    
    def __init__(self, num_actions: int, config: Dict[str, Any]):
        """
        Initialize A2C model
        
        Args:
            num_actions: Number of possible actions
            config: Model configuration dictionary
        """
        super().__init__('a2c_model')
        
        self.num_actions = num_actions
        self.config = config
        
        # CNN layers for feature extraction
        self.conv1 = kl.Convolution2D(
            filters=config['filter_no'][0],
            kernel_size=config['kernel_size'][0],
            padding="same",
            activation='relu',
            name='conv1'
        )
        
        self.conv2 = kl.Convolution2D(
            filters=config['filter_no'][1],
            kernel_size=config['kernel_size'][1],
            strides=config['strides'][1],
            padding="same",
            activation='relu',
            name='conv2'
        )
        
        self.flatten = kl.Flatten(name='flatten')
        
        # Separate hidden layers for actor and critic
        self.actor_hidden = kl.Dense(
            config['hidden_size'][0],
            activation='relu',
            name='actor_hidden'
        )
        
        self.critic_hidden = kl.Dense(
            config['hidden_size'][1],
            activation='relu',
            name='critic_hidden'
        )
        
        # Output layers
        self.value_head = kl.Dense(1, name='value')
        self.policy_head = kl.Dense(num_actions, name='policy_logits')
        
        # Probability distribution for action sampling
        self.dist = ProbabilityDistribution()
        
        self.logger = logging.getLogger(__name__)
    
    def call(self, inputs):
        """
        Forward pass through the network
        
        Args:
            inputs: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tuple of (policy_logits, value_estimates)
        """
        # Convert inputs to tensor if needed
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        
        # Separate paths for actor and critic
        actor_features = self.actor_hidden(x)
        critic_features = self.critic_hidden(x)
        
        # Output heads
        policy_logits = self.policy_head(actor_features)
        value_estimates = self.value_head(critic_features)
        
        return policy_logits, value_estimates
    
    def action_value(self, observation):
        """
        Get action and value estimate for a single observation
        
        Args:
            observation: Single observation
            
        Returns:
            Tuple of (action, value_estimate)
        """
        # Add batch dimension if needed
        if len(observation.shape) == 3:
            observation = np.expand_dims(observation, axis=0)
            
        logits, value = self.predict(observation)
        action = self.dist.predict(logits)
        
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) Agent
    
    Implements the A2C algorithm with experience collection,
    advantage estimation, and policy/value function updates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize A2C agent
        
        Args:
            config: Configuration dictionary containing hyperparameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Hyperparameters
        self.gamma = config['training']['gamma']
        self.entropy_coeff = config['training']['entropy_coefficient']
        self.value_coeff = config['training']['value_coefficient']
        self.learning_rate = config['training']['learning_rate']
        
        # Create model
        self.model = A2CModel(
            num_actions=config['environment']['action_dim'],
            config=config['model']
        )
        
        # Compile model with custom losses
        self.model.compile(
            optimizer=ko.RMSprop(learning_rate=self.learning_rate),
            loss=[self._policy_loss, self._value_loss]
        )
        
        self.logger.info("A2C Agent initialized")
    
    def get_action(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Get action and value estimate for given observation
        
        Args:
            observation: Current observation
            
        Returns:
            Tuple of (action, value_estimate)
        """
        action, value = self.model.action_value(observation)
        return int(action), float(value)
    
    def train_step(self, observations: np.ndarray, actions: np.ndarray, 
                   returns: np.ndarray, advantages: np.ndarray) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            observations: Batch of observations
            actions: Batch of actions taken
            returns: Batch of discounted returns
            advantages: Batch of advantage estimates
            
        Returns:
            Dictionary of training metrics
        """
        # Combine actions and advantages for the policy loss
        actions_and_advantages = np.concatenate([
            actions[:, None], advantages[:, None]
        ], axis=-1)
        
        # Train the model
        losses = self.model.train_on_batch(
            observations,
            [actions_and_advantages, returns]
        )
        
        return {
            'total_loss': losses[0],
            'policy_loss': losses[1],
            'value_loss': losses[2]
        }
    
    def compute_returns_and_advantages(self, rewards: np.ndarray, 
                                     values: np.ndarray, 
                                     dones: np.ndarray,
                                     next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and advantages
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for next state
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Compute returns (discounted sum of future rewards)
        returns[-1] = rewards[-1] + self.gamma * next_value * (1 - dones[-1])
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        
        # Compute advantages (returns - baseline)
        advantages = returns - values
        
        return returns, advantages
    
    def _policy_loss(self, actions_and_advantages, logits):
        """
        Compute policy loss with entropy regularization
        
        Args:
            actions_and_advantages: Combined actions and advantages
            logits: Policy logits
            
        Returns:
            Policy loss tensor
        """
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)
        
        # Policy loss (negative log probability weighted by advantages)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        
        # Entropy loss for exploration
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        
        # Combined loss (note: signs are flipped because optimizer minimizes)
        return policy_loss - self.entropy_coeff * entropy_loss
    
    def _value_loss(self, returns, values):
        """
        Compute value function loss
        
        Args:
            returns: Target returns
            values: Predicted values
            
        Returns:
            Value loss tensor
        """
        return self.value_coeff * kls.mean_squared_error(returns, values)
    
    def save_model(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath)
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        return self.model.summary()
