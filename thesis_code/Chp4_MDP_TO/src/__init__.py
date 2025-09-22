"""
MDP Topology Optimizer - Source Package

This package contains the core modules for topology optimization of electromagnetic devices:
- Q-Learning agent and trainer
- Genetic Algorithm implementation
- Environment definitions for SynRM and C-core
- Physics simulation utilities
- Visualization tools
"""

__version__ = "1.0.0"
__author__ = "Thesis Project"
__description__ = "MDP-based Topology Optimization for Electromagnetic Devices"

# Package-level imports
from . import q_learning
from . import genetic_algorithm
from . import environments
from . import utils

__all__ = [
    'q_learning',
    'genetic_algorithm',
    'environments',
    'utils'
]
