"""
Constants for FEMM-based C-core optimization environment
Based on the original A2C implementation parameters
"""

# Machine and Path Configuration
MACHINE = 'CcoreFemm'
FEMM_PATH = r'C:\femm42'  # FEMM installation path
MODEL_PATH = 'results/models'
TRAIN_PATH = 'results/data'

# Environment Dimensions
MAX_STEPS = 75
ENV_DIM = [18, 35]  # [rows, columns]
ACTION_DIM = 8
STATE_SIZE = [8, 15, 22]  # 4 previous states and 2 channels
MAX_IRON = 190

# Physics and Simulation Parameters
PENALTY = -10.0
GAMMA = 0.95  # Discount factor
LR = 0.0007  # Learning rate
BATCH_SIZE = 1024
MINI_BATCH_SIZE = 512

# Neural Network Architecture
HIDDEN_SIZE = [256, 64]
KERNEL_SIZE = [3, 3]
FILTER_NO = [8, 16]
STRIDES = [1, 2]

# Training Parameters
MAX_EPISODES = 500
WARMUP_STEPS = 500
EXPERIENCE_LENGTH = 2500
PLAY_TIME = 2

# Exploration Parameters
MAX_EPS = 1.0
MIN_EPS = 0.01
DECAY_RATE = 0.001

# Loss Function Parameters
ENTROPY = 0.001
VALUE_MULTIPLIER = 0.5

# PPO Specific Parameters (for future use)
PPO_EPOCHS = 10
CLIP_RATIO = 0.2
TARGET_KL = 0.01
GAE_LAMBDA = 0.95

# DQN Specific Parameters (for future use)
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 1000
BUFFER_SIZE = 100000
DOUBLE_DQN = True
DUELING_DQN = True

# Material Properties (for FEMM)
MATERIALS = {
    'air': {'mu': 1.0, 'sigma': 0.0},
    'iron': {'mu': 2100.0, 'sigma': 0.0},
    'copper': {'mu': 1.0, 'sigma': 58000000.0}
}

# Action Space Definition
# 0: RIGHT (1 step), 1: LEFT (1 step), 2: UP (1 step), 3: DOWN (1 step)
# 4: RIGHT (3 steps), 5: LEFT (3 steps), 6: UP (3 steps), 7: DOWN (3 steps)
ACTIONS = {
    0: {'name': 'RIGHT_1', 'dx': 0, 'dy': 1},
    1: {'name': 'LEFT_1', 'dx': 0, 'dy': -1},
    2: {'name': 'UP_1', 'dx': -1, 'dy': 0},
    3: {'name': 'DOWN_1', 'dx': 1, 'dy': 0},
    4: {'name': 'RIGHT_3', 'dx': 0, 'dy': 3},
    5: {'name': 'LEFT_3', 'dx': 0, 'dy': -3},
    6: {'name': 'UP_3', 'dx': -3, 'dy': 0},
    7: {'name': 'DOWN_3', 'dx': 3, 'dy': 0}
}

# Action Flip Dictionary (for state augmentation)
ACTION_FLIP = {
    0: 1,  # RIGHT_1 -> LEFT_1
    1: 0,  # LEFT_1 -> RIGHT_1
    2: 2,  # UP_1 -> UP_1
    3: 3,  # DOWN_1 -> DOWN_1
    4: 5,  # RIGHT_3 -> LEFT_3
    5: 4,  # LEFT_3 -> RIGHT_3
    6: 6,  # UP_3 -> UP_3
    7: 7   # DOWN_3 -> DOWN_3
}

# Geometry Parameters
GEOMETRY = {
    'coil_start': 2,
    'coil_width': 3,
    'coil_length': 9,
    'arm_start': 27,
    'arm_width': 6,
    'arm_length': 15,
    'window_width': 21,
    'window_length': 15,
    'buffer': 3
}

# Material IDs for FEMM
MATERIAL_IDS = {
    'air': 0,
    'iron': 1,
    'copper': 2,
    'pointer': 5,
    'boundary': 3
}
