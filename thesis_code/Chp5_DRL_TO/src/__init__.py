"""
C-core Reinforcement Learning Optimization Framework

A unified framework for electromagnetic C-core actuator optimization using
multiple reinforcement learning algorithms including A2C, PPO, and DQN.
"""

from .femm_environment import CcoreFemmEnv
from .a2c import A2CAgent, A2CTrainer

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    'CcoreFemmEnv',
    'A2CAgent', 
    'A2CTrainer'
]
