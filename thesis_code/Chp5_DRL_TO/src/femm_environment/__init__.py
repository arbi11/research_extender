"""
FEMM-based C-core optimization environment for reinforcement learning
"""

from .constants import *
from .environment import CcoreFemmEnv

__all__ = ['CcoreFemmEnv', 'constants']
