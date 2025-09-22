# C-core Reinforcement Learning Optimization

A unified framework for electromagnetic C-core actuator optimization using multiple reinforcement learning algorithms including A2C, PPO, and DQN with FEMM simulation.

## Overview

This project provides a comprehensive reinforcement learning framework for optimizing the topology of C-core electromagnetic actuators. The framework integrates with FEMM (Finite Element Method Magnetics) to provide accurate electromagnetic simulations and uses state-of-the-art RL algorithms to learn optimal material placement strategies.

## Features

- **Multiple RL Algorithms**: A2C (Advantage Actor-Critic), PPO (Proximal Policy Optimization), DQN (Deep Q-Network)
- **FEMM Integration**: Real-time electromagnetic simulation using FEMM
- **CNN-based State Representation**: Convolutional neural networks for processing 2D material distributions
- **Comprehensive Logging**: TensorBoard integration and detailed training metrics
- **Modular Design**: Clean separation of algorithms, environment, and training logic
- **Professional Configuration**: YAML-based configuration management

## Project Structure

```
c_core_rl_optimization/
├── main.py                    # Main entry point
├── config_a2c.yaml           # A2C configuration
├── config_ppo.yaml           # PPO configuration  
├── config_dqn.yaml           # DQN configuration
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── femm_environment/
    │   ├── __init__.py
    │   ├── constants.py      # Environment constants
    │   └── environment.py    # FEMM-based C-core environment
    ├── a2c/
    │   ├── __init__.py
    │   ├── agent.py          # A2C agent implementation
    │   └── trainer.py        # A2C training logic
    ├── ppo/                  # PPO implementation (TODO)
    │   ├── __init__.py
    │   ├── agent.py
    │   └── trainer.py
    └── dqn/                  # DQN implementation (TODO)
        ├── __init__.py
        ├── agent.py
        └── trainer.py
```

## Installation

### Prerequisites

1. **FEMM (Finite Element Method Magnetics)**
   - Download and install FEMM from: http://www.femm.info/
   - Default installation path: `C:\femm42`
   - Update the `femm_path` in configuration files if installed elsewhere

2. **Python Dependencies**
   ```bash
   pip install tensorflow numpy pandas scipy matplotlib pyyaml imageio scikit-image
   ```

3. **Additional Requirements**
   - Windows OS (required for FEMM integration)
   - Python 3.7+ with TensorFlow 2.x

### Setup

1. Navigate to the project directory:
   ```bash
   cd thesis_code/Chp5_RL/polished_version
   ```

2. Verify FEMM installation:
   - Ensure FEMM is installed and accessible
   - Update `femm_path` in config files if needed

## Usage

### Quick Start

Run A2C training with default configuration:
```bash
python main.py --method a2c
```

### Command Line Options

```bash
python main.py --method {a2c,ppo,dqn} [OPTIONS]

Options:
  --method {a2c,ppo,dqn}    RL algorithm to use (required)
  --config CONFIG_FILE      Custom configuration file path
  --results-dir RESULTS     Custom results directory
  --verbose                 Enable verbose logging
  --help                    Show help message
```

### Examples

1. **A2C with default settings:**
   ```bash
   python main.py --method a2c
   ```

2. **A2C with custom configuration:**
   ```bash
   python main.py --method a2c --config my_config.yaml
   ```

3. **A2C with verbose logging:**
   ```bash
   python main.py --method a2c --verbose
   ```

4. **A2C with custom results directory:**
   ```bash
   python main.py --method a2c --results-dir ./my_results
   ```

## Configuration

### A2C Configuration (`config_a2c.yaml`)

```yaml
# Environment Configuration
environment:
  name: "CcoreFemmEnv"
  max_steps: 75              # Maximum steps per episode
  env_dim: [18, 35]          # Environment dimensions
  action_dim: 8              # Number of actions (8 directions)
  state_size: [8, 15, 22]    # State representation size
  max_iron: 190              # Maximum iron elements
  penalty: -10.0             # Penalty for invalid actions

# Neural Network Architecture
model:
  hidden_size: [256, 64]     # Hidden layer sizes
  kernel_size: [3, 3]        # CNN kernel sizes
  filter_no: [8, 16]         # CNN filter numbers
  strides: [1, 2]            # CNN strides

# Training Configuration
training:
  learning_rate: 0.0007      # Learning rate
  batch_size: 1024           # Batch size for training
  gamma: 0.95                # Discount factor
  entropy_coefficient: 0.001 # Entropy regularization
  value_coefficient: 0.5     # Value loss weight
  max_episodes: 500          # Maximum training episodes
  max_updates: 500           # Maximum training updates

# Paths
paths:
  femm_path: "C:\\femm42"    # FEMM installation path
  model_path: "results/models"
  log_path: "results/logs"
```

## Environment Details

### C-core FEMM Environment

The environment simulates a C-core electromagnetic actuator with:

- **Design Space**: 2D grid where the agent places magnetic material
- **Actions**: 8 directional movements (4 single-step + 4 triple-step)
- **State**: 2-channel representation (material distribution + magnetic field)
- **Reward**: Based on magnetic field strength in the armature region
- **Physics**: Real-time FEMM electromagnetic simulation

### Action Space

- **0-3**: Single-step movement (Right, Left, Up, Down)
- **4-7**: Triple-step movement (Right, Left, Up, Down)

### State Representation

- **Channel 1**: Material distribution (air=0, iron=1, copper=2, boundary=3, pointer=5)
- **Channel 2**: Magnetic field distribution (B-field magnitude)
- **Shape**: [2, 15, 22] (2 channels, 15 rows, 22 columns)

## Algorithm Details

### A2C (Advantage Actor-Critic)

- **Architecture**: CNN feature extraction + separate actor-critic heads
- **Loss Function**: Policy gradient + value function + entropy regularization
- **Advantage Estimation**: Temporal difference learning
- **Experience Collection**: On-policy batch collection

**Key Features:**
- Convolutional layers for spatial feature extraction
- Separate networks for policy and value estimation
- Entropy regularization for exploration
- Batch training with advantage estimation

### PPO (Proximal Policy Optimization) - TODO

Planned features:
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Multiple epochs per batch
- KL divergence early stopping

### DQN (Deep Q-Network) - TODO

Planned features:
- Experience replay buffer
- Target network updates
- Double DQN and Dueling DQN options
- Epsilon-greedy exploration

## Output and Results

### Directory Structure

After training, results are saved in timestamped directories:

```
results/
└── 2025_Sep20_2230_A2C/
    ├── training.log           # Training logs
    ├── main.log              # Main execution logs
    ├── training_results.json # Comprehensive results
    ├── checkpoints/          # Model checkpoints
    │   ├── model_ep10_up50.h5
    │   └── model_ep20_up100.h5
    ├── renders/              # Episode renderings
    │   ├── Eps_1/
    │   └── Eps_2/
    └── eval_renders/         # Evaluation renderings
```

### Training Metrics

The framework tracks comprehensive metrics:

- **Episode Metrics**: Reward, length, iron count, net force
- **Training Metrics**: Policy loss, value loss, advantages
- **Evaluation Metrics**: Periodic performance evaluation
- **Physics Metrics**: Magnetic field strength, force calculations

### Results Analysis

Training results include:

```json
{
  "training_time": 1234.56,
  "total_episodes": 500,
  "total_updates": 500,
  "episode_rewards": [...],
  "final_performance": {
    "mean_reward_last_10": 45.67,
    "best_reward": 89.12,
    "final_evaluation": {...}
  }
}
```

## Extending the Framework

### Adding New Algorithms

1. Create algorithm directory: `src/new_algorithm/`
2. Implement agent class with required interface
3. Implement trainer class following existing patterns
4. Add configuration file: `config_new_algorithm.yaml`
5. Update main.py to include new algorithm

### Customizing the Environment

The FEMM environment can be customized by:

- Modifying geometry parameters in `constants.py`
- Adjusting reward functions in `environment.py`
- Changing state representation
- Adding new physics constraints

### Configuration Customization

All hyperparameters can be tuned via YAML configuration files:

- Network architecture parameters
- Training hyperparameters
- Environment settings
- Logging and visualization options

## Troubleshooting

### Common Issues

1. **FEMM Not Found**
   ```
   Error: FEMM path not found: C:\femm42
   ```
   **Solution**: Install FEMM or update `femm_path` in config

2. **Memory Issues**
   ```
   Error: Out of memory during training
   ```
   **Solution**: Reduce `batch_size` or `max_episodes` in config

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'src'
   ```
   **Solution**: Run from project root directory

4. **FEMM COM Interface Issues**
   ```
   Error: FEMM COM interface failed
   ```
   **Solution**: Ensure FEMM is properly installed and registered

### Debug Mode

Enable verbose logging for debugging:
```bash
python main.py --method a2c --verbose
```

### Performance Optimization

For better performance:
- Use GPU-enabled TensorFlow
- Adjust batch sizes based on available memory
- Reduce rendering frequency during training
- Use faster FEMM solver settings

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{ccore_rl_optimization,
  title={C-core Reinforcement Learning Optimization Framework},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions and support:

- Check the troubleshooting section
- Review configuration examples
- Open an issue on GitHub
- Contact the maintainers

## Acknowledgments

- FEMM development team for the electromagnetic simulation software
- TensorFlow team for the deep learning framework
- Original A2C implementation authors
- Electromagnetic design optimization research community
