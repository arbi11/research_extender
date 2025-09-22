# MDP Topology Optimizer

A comprehensive framework for topology optimization of electromagnetic devices using reinforcement learning (Q-Learning) and evolutionary algorithms (Genetic Algorithm). This system supports two environments:

1. **SynRM (Synchronous Reluctance Motor)** - Rotor geometry optimization
2. **C-core** - C-shaped magnetic core optimization

## Features

- **Dual Environment Support**: Optimized for both SynRM and C-core electromagnetic devices
- **Multiple Optimization Methods**: Q-Learning and Genetic Algorithm implementations
- **Pure Python Implementation**: No external dependencies on MATLAB or commercial software
- **Configurable Physics**: Simplified but meaningful electromagnetic simulation
- **Comprehensive Logging**: Detailed training progress and results tracking
- **Visualization Tools**: Generate topology evolution plots and performance graphs
- **Repository Ready**: Clean, modular code structure suitable for sharing

## Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager

### Install Dependencies
```bash
# Navigate to the project directory
cd thesis_code/Chp4_MDP/polished_version

# Install from pyproject.toml
pip install -e .
```

### Manual Installation
If you prefer to install dependencies manually:
```bash
pip install numpy scipy matplotlib pyyaml scikit-learn scikit-image imageio Pillow deap seaborn tqdm
```

## Quick Start

### Basic Usage

#### Q-Learning on SynRM Environment
```bash
python main.py --method ql --environment synrm
```

#### Genetic Algorithm on C-core Environment
```bash
python main.py --method ga --environment ccore
```

#### Compare Both Methods
```bash
python main.py --method compare
```

### Configuration

The system uses a YAML configuration file (`config.yaml`) for all parameters. Key sections:

#### Q-Learning Parameters
```yaml
q_learning:
  episodes: 100          # Number of training episodes
  learning_rate: 0.8     # Learning rate for Q-value updates
  gamma: 0.95           # Discount factor
  epsilon_decay: 0.05   # Exploration decay rate
  max_steps: 15         # Maximum steps per episode
```

#### Genetic Algorithm Parameters
```yaml
genetic_algorithm:
  population_size: 100   # Population size
  generations: 50       # Number of generations
  crossover_prob: 0.8   # Crossover probability
  mutation_prob: 0.1    # Mutation probability
```

#### Environment Parameters
```yaml
environments:
  synrm:
    grid_sizes: [6, 7, 8, 9, 10]  # Supported grid sizes
    max_flux_barriers: 3          # Maximum flux barriers
    rotor_outer_diameter: 100     # Rotor dimensions
    rotor_inner_diameter: 50

  ccore:
    grid_sizes: [3, 4, 5]         # Supported grid sizes
    core_width: 20                # Core dimensions
    core_height: 30
    air_gap: 2
```

## Project Structure

```
polished_version/
├── main.py                    # Main entry point with CLI
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── src/
    ├── __init__.py          # Package initialization
    ├── q_learning/          # Q-Learning implementation
    │   ├── __init__.py
    │   ├── agent.py         # Q-Learning agent
    │   └── trainer.py       # Training coordinator
    ├── genetic_algorithm/   # Genetic Algorithm implementation
    │   ├── __init__.py
    │   ├── algorithm.py     # GA implementation
    │   └── objective.py     # Fitness function
    ├── environments/        # Environment definitions
    │   ├── __init__.py
    │   ├── base_env.py      # Base environment class
    │   ├── synrm_env.py     # SynRM environment
    │   └── ccore_env.py     # C-core environment
    └── utils/               # Utility functions
        ├── __init__.py
        ├── physics.py       # Physics simulation
        └── visualization.py # Plotting utilities
```

## Detailed Usage

### Q-Learning

The Q-Learning implementation uses:
- **Dictionary-based Q-table** for sparse state space
- **Epsilon-greedy exploration** strategy
- **Experience replay** for stable learning
- **Topology tracking** for best design identification

```python
from src.q_learning.trainer import QLearningTrainer
from environments.synrm_env import SynRMEnvironment

# Load configuration
config = load_config('config.yaml')

# Create environment and trainer
env = SynRMEnvironment(config)
trainer = QLearningTrainer(env, config)

# Train the agent
trainer.train(results_dir="results/synrm_ql")

# Test the trained agent
test_results = trainer.test_agent(results_dir="results/synrm_ql", num_episodes=10)
```

### Genetic Algorithm

The Genetic Algorithm implementation uses:
- **DEAP framework** for robust evolutionary computation
- **Binary representation** for topology encoding
- **Tournament selection** for parent selection
- **Custom fitness function** based on electromagnetic performance

```python
from src.genetic_algorithm.algorithm import GeneticAlgorithm
from environments.ccore_env import CCoreEnvironment

# Load configuration
config = load_config('config.yaml')

# Create environment and GA
env = CCoreEnvironment(config)
ga = GeneticAlgorithm(env, config)

# Run optimization
best_topology, best_fitness = ga.optimize(results_dir="results/ccore_ga")
```

## Physics Simulation

The system includes simplified but meaningful electromagnetic simulation:

### SynRM Physics
- **Saliency ratio** calculation
- **Flux barrier effectiveness** evaluation
- **Torque production** estimation
- **Material efficiency** scoring

### C-core Physics
- **Magnetic path efficiency** analysis
- **Field concentration** evaluation
- **Structural integrity** assessment
- **Material usage** optimization

## Results and Output

### Training Results
Each run creates a timestamped results directory containing:
- `training.log` - Detailed training logs
- `training_history.json` - Complete training statistics
- `final_agent.npz` - Trained Q-Learning agent (if applicable)
- `best_topology.json` - Best topology found
- `agent_statistics.json` - Agent performance metrics

### Visualization
The system generates various plots:
- **Topology evolution** over training episodes
- **Reward curves** showing learning progress
- **Best topology** visualization
- **Performance comparisons** between methods

## Configuration Details

### Physics Parameters
```yaml
physics:
  synrm:
    saliency_weight: 0.4        # Weight for saliency ratio
    torque_weight: 0.4          # Weight for torque production
    efficiency_weight: 0.2      # Weight for material efficiency
    connectivity_threshold: 0.6 # Minimum connectivity threshold

  ccore:
    connectivity_weight: 0.5     # Weight for magnetic connectivity
    field_concentration_weight: 0.3  # Weight for field concentration
    material_efficiency_weight: 0.2  # Weight for material usage
    min_connected_components: 1 # Minimum connected components
```

### Training Parameters
```yaml
training:
  save_frequency: 10    # Save checkpoint every N episodes
  results_dir: "results"  # Base results directory
  log_level: "INFO"     # Logging level
  random_seed: 42       # Random seed for reproducibility
```

## Examples and Tutorials

### Example 1: Quick SynRM Optimization
```bash
# Run Q-Learning on SynRM for 50 episodes
python main.py --method ql --environment synrm
```

### Example 2: Comprehensive C-core Study
```bash
# Run both methods on C-core and compare results
python main.py --method compare
```

### Example 3: Custom Configuration
```python
# Modify config.yaml to change parameters
# Then run with custom settings
python main.py --method ga --environment ccore
```

## Performance Tips

1. **Grid Size**: Start with smaller grid sizes (3x3, 4x4) for faster training
2. **Episodes**: Use 50-100 episodes for initial testing
3. **Population Size**: Start with smaller populations (50-100) for GA
4. **Hardware**: GPU acceleration not required but can speed up large grid training

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Memory Issues**: Reduce grid size or population size for large problems
3. **Convergence Problems**: Adjust learning rates or exploration parameters
4. **Poor Performance**: Check physics parameters and reward function weights

### Debug Mode
Enable debug logging by modifying the config:
```yaml
training:
  log_level: "DEBUG"
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mdp_topology_optimizer,
  title={MDP Topology Optimizer: Q-Learning and GA for Electromagnetic Design},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo/mdp-topology-optimizer}}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Q-Learning implementation inspired by electromagnetic topology optimization research
- DEAP library for genetic algorithm framework
- OpenAI Gym for environment design patterns
- Electromagnetic simulation based on fundamental physics principles

---

**Note**: This is a research implementation focused on demonstrating the application of reinforcement learning and evolutionary algorithms to electromagnetic topology optimization. The physics simulation is simplified for computational efficiency while maintaining the essential characteristics of the optimization problem.
