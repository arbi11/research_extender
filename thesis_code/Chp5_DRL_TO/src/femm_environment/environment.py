"""
FEMM-based C-core optimization environment for reinforcement learning
Cleaned up and modularized version of the original A2C environment
"""

import os
import numpy as np
import femm
import imageio
import warnings
from skimage import img_as_float, img_as_uint
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .constants import *


class CcoreFemmEnv:
    """
    C-core electromagnetic actuator optimization environment using FEMM

    This environment allows RL agents to optimize the topology of a C-core
    electromagnetic actuator by placing magnetic material in a design space.
    The reward is based on the magnetic field strength in the armature region.
    """

    def __init__(self,
                 max_steps: int = MAX_STEPS,
                 env_dim: List[int] = None,
                 start_position: Tuple[int, int] = (3, 4),
                 femm_path: str = FEMM_PATH,
                 render_path: str = MODEL_PATH,
                 log_level: str = "INFO"):
        """
        Initialize the C-core FEMM environment

        Args:
            max_steps: Maximum steps per episode
            env_dim: Environment dimensions [rows, cols]
            start_position: Starting position of the design pointer
            femm_path: Path to FEMM installation
            render_path: Path for saving renderings
            log_level: Logging level
        """

        # Environment configuration
        self.max_steps = max_steps
        self.env_dim = env_dim or ENV_DIM
        self.start_posR, self.start_posC = start_position

        # FEMM and file paths
        self.femm_path = Path(femm_path)
        self.render_path = Path(render_path)
        self.render_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)

        # Episode tracking
        self.frame_idx = 0
        self.step_count = 0
        self.done = False

        # State tracking
        self.current_state = None
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.force_history = []
        self.bfield_history = []

        # Position and movement
        self.posR = self.start_posR
        self.posC = self.start_posC
        self.worm_step_size = 3

        # Physics state
        self.net_force = 0.0
        self.bfield_state = np.zeros([15,])
        self.bfield_xy = np.zeros([self.env_dim[0] - GEOMETRY['buffer'], self.env_dim[1]])
        self.mu_state = np.zeros([self.env_dim[0] - GEOMETRY['buffer'], self.env_dim[1]])

        # State augmentation
        self.flip = 0  # Random flip for data augmentation
        self.flip_actions = ACTION_FLIP

        # Geometry parameters
        self.geometry = GEOMETRY.copy()

        # FEMM document
        self.femm_doc = None

        self.logger.info("C-core FEMM environment initialized")

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state

        Returns:
            Initial state observation
        """

        # Reset episode tracking
        self.frame_idx += 1
        self.step_count = 0
        self.done = False

        # Reset histories
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.force_history = []
        self.bfield_history = []

        # Reset position
        self.posR = self.start_posR
        self.posC = self.start_posC
        self.worm_step_size = 3

        # Reset physics
        self.net_force = 0.0
        self.bfield_state = np.zeros([15,])
        self.bfield_xy = np.zeros([self.env_dim[0] - self.geometry['buffer'], self.env_dim[1]])
        self.mu_state = np.zeros([self.env_dim[0] - self.geometry['buffer'], self.env_dim[1]])

        # Random flip for data augmentation
        self.flip = np.random.randint(2)

        # Create initial geometry
        self._create_geometry()

        # Create initial state
        initial_state = self._create_initial_state()

        # Store first state
        self.state_history.append(initial_state.copy())
        self.action_history.append(4)  # Default initial action

        # Calculate initial reward
        initial_reward = self._calculate_reward(initial_state)
        self.reward_history.append(initial_reward)

        # Save first frame
        self._save_frame(0, "reset")

        self.logger.info(f"Environment reset - Episode {self.frame_idx}")
        return self._get_observation(initial_state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step

        Args:
            action: Action to take (0-7)

        Returns:
            Tuple of (next_state, reward, done, info)
        """

        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")

        self.step_count += 1

        # Get current state
        current_state = self.state_history[-1].copy()

        # Execute action
        new_state, issue = self._execute_action(current_state, action)

        # Calculate reward
        if issue == 0:
            reward = self._calculate_reward(new_state)
            self.reward_history.append(reward)
        else:
            # Penalty for invalid actions
            reward = PENALTY
            self.reward_history.append(reward)

        # Check termination conditions
        self.done = self._check_termination(new_state)

        # Store new state
        self.state_history.append(new_state.copy())
        self.action_history.append(action)

        # Get observation
        observation = self._get_observation(new_state)

        # Info dictionary
        info = {
            'step': self.step_count,
            'issue': issue,
            'iron_count': np.sum(new_state == MATERIAL_IDS['iron']),
            'net_force': self.net_force,
            'reward': reward
        }

        self.logger.debug(f"Step {self.step_count}: action={action}, reward={reward:.2f}, done={self.done}")
        return observation, reward, self.done, info

    def _create_geometry(self) -> None:
        """Create the base FEMM geometry with coils and armature"""

        try:
            # Start FEMM
            femm.openfemm(1)
            femm.opendocument(str(self.femm_path / 'actuator.fem'))

            # Define materials
            femm.mi_getmaterial('Air')
            femm.mi_getmaterial('Cold rolled low carbon strip steel')
            femm.mi_getmaterial('Copper')
            femm.mi_addcircprop('icoil', 1, 1)

            # Clear existing labels
            for i in range(17):
                for j in range(35):
                    femm.mi_selectlabel(i + 0.5, j + 0.5)

            femm.mi_setblockprop('Air', 0, 0.5, '<None>', 0, 0, 0)
            femm.mi_clearselected()

            # Create coils
            self._create_coils()

            # Create armature
            self._create_armature()

            # Create boundaries
            self._create_boundaries()

            # Save geometry
            femm.mi_zoomnatural()
            femm.mi_saveas('actuator2.fem')
            femm.closefemm()

        except Exception as e:
            self.logger.error(f"Error creating geometry: {e}")
            raise

    def _create_coils(self) -> None:
        """Create the electromagnetic coils"""

        # Left coil
        femm.mi_selectrectangle(0.1, self.geometry['coil_start'] + 0.1,
                               self.geometry['coil_length'] - 0.1,
                               self.geometry['coil_start'] + self.geometry['coil_width'] - 0.1, 4)
        femm.mi_deleteselectedlabels()
        femm.mi_deleteselectednodes()
        femm.mi_deleteselectedsegments()
        femm.mi_drawrectangle(0, self.geometry['coil_start'],
                             self.geometry['coil_length'],
                             self.geometry['coil_start'] + self.geometry['coil_width'])
        femm.mi_addblocklabel(self.geometry['coil_length'] / 2,
                             (2 * self.geometry['coil_start'] + self.geometry['coil_width']) / 2)
        femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, -500)

        # Right coil
        coil2_start = 8 + 8 - (self.geometry['coil_start'] + self.geometry['coil_width'])
        femm.mi_selectrectangle(0.1, coil2_start + 0.1,
                               self.geometry['coil_length'] - 0.1,
                               coil2_start + self.geometry['coil_width'] - 0.1, 4)
        femm.mi_deleteselectedlabels()
        femm.mi_deleteselectednodes()
        femm.mi_deleteselectedsegments()
        femm.mi_drawrectangle(0, coil2_start, self.geometry['coil_length'],
                             coil2_start + self.geometry['coil_width'])
        femm.mi_addblocklabel(self.geometry['coil_length'] / 2,
                             (2 * coil2_start + self.geometry['coil_width']) / 2)
        femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, 500)

    def _create_armature(self) -> None:
        """Create the magnetic armature"""

        femm.mi_selectrectangle(0.1, self.geometry['arm_start'] + 0.1,
                               self.geometry['arm_length'] - 0.1,
                               self.geometry['arm_start'] + self.geometry['arm_width'] - 0.1, 4)
        femm.mi_deleteselectedlabels()
        femm.mi_deleteselectednodes()
        femm.mi_deleteselectedsegments()
        femm.mi_drawrectangle(0, self.geometry['arm_start'],
                             self.geometry['arm_length'],
                             self.geometry['arm_start'] + self.geometry['arm_width'])
        femm.mi_addblocklabel(self.geometry['arm_length'] / 2,
                             (2 * self.geometry['arm_start'] + self.geometry['arm_width']) / 2)
        femm.mi_setblockprop('Cold rolled low carbon strip steel', 0, 0.5, '<None>', 0, 5, 0)

    def _create_boundaries(self) -> None:
        """Create boundary conditions"""

        # Outer boundary
        femm.mi_drawrectangle(0, 100, 100, -100)
        femm.mi_addblocklabel(50, 0)
        femm.mi_setblockprop('Air', 50, 0, '<None>', 0, 0, 0)

        # Boundary properties
        femm.mi_addboundprop('neumann', 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0)
        femm.mi_selectrectangle(-0.5, 100.5, 0.2, -100.5, 4)
        femm.mi_setsegmentprop('neumann', 0, 0, 0, 5)

        femm.mi_selectsegment(95, -95)
        femm.mi_selectsegment(55, 95)
        femm.mi_selectsegment(55, -95)
        femm.mi_addboundprop('dirichlet', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_setsegmentprop('dirichlet', 0, 0, 0, 0)

    def _create_initial_state(self) -> np.ndarray:
        """Create the initial state with design space"""

        # Create base state
        state = np.ones(self.env_dim) * MATERIAL_IDS['boundary']

        # Create design window
        state[self.geometry['buffer']:self.geometry['buffer'] + self.geometry['window_length'],
              self.geometry['coil_start'] + self.geometry['coil_width']:
              self.geometry['coil_start'] + self.geometry['coil_width'] + self.geometry['window_width']] = MATERIAL_IDS['air']

        # Add coils
        state[self.geometry['buffer']:self.geometry['buffer'] + self.geometry['coil_length'],
              self.geometry['coil_start']:self.geometry['coil_start'] + self.geometry['coil_width']] = MATERIAL_IDS['copper']

        coil2_start = 8 + 8 - (self.geometry['coil_start'] + self.geometry['coil_width'])
        state[self.geometry['buffer']:self.geometry['buffer'] + self.geometry['coil_length'],
              coil2_start:coil2_start + self.geometry['coil_width']] = MATERIAL_IDS['copper']

        # Add armature
        state[self.geometry['buffer']:self.geometry['buffer'] + self.geometry['arm_length'],
              self.geometry['arm_start']:self.geometry['arm_start'] + self.geometry['arm_width']] = MATERIAL_IDS['iron']

        return state

    def _execute_action(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, int]:
        """
        Execute an action and return new state and issue code

        Args:
            state: Current state
            action: Action to execute

        Returns:
            Tuple of (new_state, issue_code)
        """

        new_state = state.copy()

        # Get action parameters
        action_info = ACTIONS[action]
        dx, dy = action_info['dx'], action_info['dy']

        # Calculate new position
        new_posR = self.posR + dx
        new_posC = self.posC + dy

        # Check bounds
        if (new_posR < 0 or new_posR >= self.env_dim[0] or
            new_posC < 0 or new_posC >= self.env_dim[1]):
            return new_state, 3  # Out of bounds

        # Check if position is valid
        if new_state[new_posR, new_posC] == MATERIAL_IDS['boundary']:
            return new_state, 2  # Boundary collision
        elif new_state[new_posR, new_posC] == MATERIAL_IDS['copper']:
            return new_state, 2  # Coil collision

        # Update position
        self.posR, self.posC = new_posR, new_posC

        # Place material (3x3 block for worm-like movement)
        r_start, r_end = max(0, self.posR - 1), min(self.env_dim[0], self.posR + 2)
        c_start, c_end = max(0, self.posC - 1), min(self.env_dim[1], self.posC + 2)

        new_state[r_start:r_end, c_start:c_end] = MATERIAL_IDS['iron']
        new_state[self.posR, self.posC] = MATERIAL_IDS['pointer']

        return new_state, 0  # Success

    def _calculate_reward(self, state: np.ndarray) -> float:
        """
        Calculate reward based on magnetic field in armature region

        Args:
            state: Current state

        Returns:
            Reward value
        """

        try:
            # Open FEMM and load geometry
            femm.openfemm(1)
            femm.opendocument('actuator2.fem')

            # Update material properties based on state
            self._update_femm_materials(state)

            # Analyze
            femm.mi_analyse(0)
            femm.mi_loadsolution()

            # Calculate magnetic field in armature
            b_sum = 0.0
            for arm in range(6):
                bx, by = femm.mo_getb(0.2, 27.5 + arm)
                b_magnitude = np.sqrt(bx**2 + by**2)
                b_sum += b_magnitude

            # Calculate force
            femm.mo_groupselectblock(5)
            self.net_force = femm.mo_blockintegral(19) * (-10)
            self.force_history.append(self.net_force)

            # Calculate B-field distribution
            self._calculate_bfield_distribution()

            femm.closefemm()

            # Reward is average B-field in armature
            reward = (b_sum / 6) * 1000

            return reward

        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return PENALTY

    def _update_femm_materials(self, state: np.ndarray) -> None:
        """Update FEMM material properties based on current state"""

        # Find changed positions
        if len(self.state_history) > 1:
            prev_state = self.state_history[-2]
            changed_positions = np.where(state != prev_state)
        else:
            changed_positions = np.where(state == MATERIAL_IDS['iron'])

        # Update materials in FEMM
        for i, j in zip(changed_positions[0], changed_positions[1]):
            if state[i, j] == MATERIAL_IDS['iron']:
                femm.mi_selectlabel(i + 0.4, j + 0.4)
                femm.mi_deleteselectedlabels()
                femm.mi_addblocklabel(i + 0.4, j + 0.4)
                femm.mi_setblockprop('Cold rolled low carbon strip steel', 0, 0, '<None>', 0, 3, 0)

    def _calculate_bfield_distribution(self) -> None:
        """Calculate B-field distribution for state representation"""

        # B-field in armature (15 points)
        self.bfield_state = np.zeros([15,])
        for row in range(15):
            self.bfield_state[row] = np.sqrt(
                femm.mo_getb(row + 0.5, 26.5)[0]**2 +
                femm.mo_getb(row + 0.5, 26.5)[1]**2
            )

        # B-field in design space
        for r in range(self.bfield_xy.shape[0]):
            for c in range(self.bfield_xy.shape[1]):
                self.bfield_xy[r, c] = np.sqrt(
                    femm.mo_getb(r + 0.5, c + 0.5)[0]**2 +
                    femm.mo_getb(r + 0.5, c + 0.5)[1]**2
                )

        # B-field magnitude squared
        self.bfield_dist = np.round(np.sum(np.square(self.bfield_xy), axis=-1), decimals=2)

    def _check_termination(self, state: np.ndarray) -> bool:
        """
        Check if episode should terminate

        Args:
            state: Current state

        Returns:
            True if episode should end
        """

        # Check iron count
        iron_count = np.sum(state == MATERIAL_IDS['iron'])
        if iron_count > MAX_IRON:
            return True

        # Check step count
        if self.step_count >= self.max_steps:
            return True

        return False

    def _get_observation(self, state: np.ndarray) -> np.ndarray:
        """
        Get observation for RL agent

        Args:
            state: Current state

        Returns:
            Observation array
        """

        # Get design window
        state_window = state[self.geometry['buffer']:self.geometry['buffer'] + 15, 5:27]

        if self.flip == 1:
            # Flip and pad for data augmentation
            state_padded = np.lib.pad(state_window, ((0, 21)), 'reflect')
            bfield_padded = np.lib.pad(self.bfield_dist[self.geometry['buffer']:self.geometry['buffer'] + 15, 5:27],
                                     ((0, 21)), 'reflect')
            observation = np.concatenate([
                np.expand_dims(state_padded, 0),
                np.expand_dims(bfield_padded, 0)
            ], axis=0)
            observation = observation[:, :15, -22:]
        else:
            # Normal observation
            observation = np.concatenate([
                np.expand_dims(state_window, 0),
                np.expand_dims(self.bfield_dist[self.geometry['buffer']:self.geometry['buffer'] + 15, 5:27], 0)
            ], axis=0)

        return observation

    def _save_frame(self, step: int, description: str) -> None:
        """Save current state as image"""

        if len(self.state_history) == 0:
            return

        state = self.state_history[-1]
        state_img = state[self.geometry['buffer']:self.geometry['buffer'] + 15, 5:26]
        state_img = state_img / np.max(state_img)

        frame_dir = self.render_path / f'Eps_{self.frame_idx}'
        frame_dir.mkdir(exist_ok=True)

        filename = f'{self.frame_idx}_Step-{step}_{description}_reward-{self.reward_history[-1]:.2f}.png'
        imageio.imwrite(str(frame_dir / filename), img_as_uint(state_img))

    def render(self, mode: str = 'human') -> None:
        """Render current state"""

        if len(self.state_history) == 0:
            return

        self._save_frame(self.step_count, "render")

    def close(self) -> None:
        """Clean up environment"""

        try:
            femm.closefemm()
        except:
            pass

        self.logger.info("Environment closed")

    @property
    def action_space(self) -> int:
        """Get action space size"""
        return ACTION_DIM

    @property
    def observation_space(self) -> Tuple[int, ...]:
        """Get observation space shape"""
        return STATE_SIZE
