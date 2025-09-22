"""
A2C (Advantage Actor-Critic) implementation for C-core optimization
"""

from .agent import A2CAgent
from .trainer import A2CTrainer

__all__ = ['A2CAgent', 'A2CTrainer']
