# Magnetic Field Predictor

A unified framework for magnetic field prediction using both Deep Learning and Physics-Informed Neural Network approaches.

## Overview

This project provides two different approaches for predicting magnetic field distributions:

1. **Deep Learning (DL)**: Uses a U-Net style CNN architecture trained on simulation data
2. **Physics-Informed Neural Network (PINN)**: Uses an Extended PINN (XPINN) that incorporates physical laws

## Project Structure

```
magnetic_field_predictor/
├── main.py                    # Main entry point
├── config_dl.yaml            # Deep Learning configuration
├── config_pinn.yaml          # PINN configuration
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── deep_learning/
    │   ├── __init__.py
    │   ├── model.py          # CNN architecture
    │   └── trainer.py        # DL training logic
    └── physics_informed/
        ├── __init__.py
        ├── xpinn_model.py    # XPINN implementation
        └── trainer.py        # PINN training logic
```

## Installation

1. Navigate to the project directory:
   ```bash
   cd thesis_code/Chp2_FieldDistribution/magnetic_field_predictor
   ```

2. Install required dependencies:
   ```bash
   pip install tensorflow numpy pandas scipy matplotlib pyyaml
   ```

## Usage

### Running Deep Learning Approach

```bash
python main.py --method dl
```

This will:
- Load configuration from `config_dl.yaml`
- Train a CNN model on magnetic field data
- Save the trained model and training metrics

### Running Physics-Informed Neural Network Approach

```bash
python main.py --method pinn
```

This will:
- Load configuration from `config_pinn.yaml`
- Train an XPINN model using physics constraints
- Save the trained model weights and predictions

## Configuration Files

### Deep Learning Configuration (`config_dl.yaml`)

```yaml
model:
  img_size: [400, 400]    # Input image dimensions
  channels: 1             # Number of input channels
  architecture: "unet_skip"  # Model architecture

training:
  batch_size: 32          # Training batch size
  epochs: 100             # Number of training epochs
  learning_rate: 0.001    # Learning rate
  optimizer: "adam"       # Optimizer type

data:
  normalization: "minmax" # Data normalization method
  augmentation: true      # Enable data augmentation
```

### PINN Configuration (`config_pinn.yaml`)

```yaml
xpinn:
  layers1: [2, 400, 400, 100, 1]  # Architecture for subdomain 1
  layers2: [2, 100, 100, 40, 1]   # Architecture for subdomain 2
  mu1: 1                          # Regularization parameter 1
  mu2: 1                          # Regularization parameter 2
  scaling_factor: 20              # Neural network scaling
  multiplier: 20                  # Loss multiplier

training:
  max_iter: 25000                 # Maximum training iterations
  adam_lr: 0.0008                 # Adam learning rate
  n_f1: 1000                      # Collocation points subdomain 1
  n_f2: 200                       # Collocation points subdomain 2
  n_ub: 500                       # Boundary points
  n_i1: 250                       # Interface points

physics:
  equation: "poisson"             # Physics equation
  domain_decomposition: "interface"  # Domain decomposition method
```

## Output

Both methods create a timestamped results directory containing:

- `training.log`: Training logs
- `training_metrics.json`: Training statistics and metrics
- `model/`: Saved model files
- `test_predictions.json`: Model predictions on test data

## Data Requirements

### Deep Learning Approach
- Requires CSV files with magnetic field simulation data
- Expected format: x, y, B_field, material_properties
- Data should be located in `data/raw/NL_Data2/` directory

### PINN Approach
- Generates synthetic training data automatically
- Uses physics constraints (Poisson equation) for training
- No external data files required

## Architecture Details

### Deep Learning Model
- U-Net style architecture with skip connections
- Batch normalization and dropout for regularization
- Designed for 400x400 input images
- Outputs magnetic field predictions

### XPINN Model
- Two separate neural networks for domain decomposition
- Physics-informed loss functions
- Interface conditions for continuity
- Solves Poisson equation: ∇²u = f(x,y)

## Extending the Framework

To add new methods:

1. Create a new subdirectory in `src/`
2. Implement model and trainer classes
3. Add configuration file
4. Update `main.py` to include the new method

## Troubleshooting

### Common Issues

1. **TensorFlow compatibility**: Ensure TensorFlow 2.x is installed
2. **Data path issues**: Check that data files exist in expected locations
3. **Memory errors**: Reduce batch size or image dimensions
4. **Training instability**: Adjust learning rates in configuration files

### Debug Mode

Add debug prints to monitor training progress:

```python
# In trainer files, add:
print(f"Step {step}: Loss = {loss:.6f}")
```

## Citation

If you use this code in your research, please cite:

```
@misc{magnetic_field_predictor,
  title={Magnetic Field Predictor: DL and PINN Approaches},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
