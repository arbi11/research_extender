# Efficiency Map Predictor

A polished, repository-ready framework for predicting efficiency maps and power factor maps using both RNN and Attention-based neural network architectures.

## Overview

This project provides two different approaches for predicting motor efficiency and power factor across operating conditions:

1. **RNN Model**: Simple GRU-based recurrent neural network
2. **Attention Model**: Advanced encoder-decoder architecture with attention mechanism

Both models can predict either:
- **Efficiency Maps**: Motor efficiency across torque-speed operating conditions
- **Power Factor Maps**: Power factor across torque-speed operating conditions

## Project Structure

```
efficiency_map_predictor/
├── main.py                    # Main entry point with CLI
├── config_rnn.yaml           # RNN model configuration
├── config_attention.yaml     # Attention model configuration
├── requirements.txt          # Dependencies (not needed with uv)
├── README.md                 # This file
└── src/
    ├── __init__.py
    ├── data_loader.py        # Unified data loading
    ├── rnn_model.py          # RNN architecture
    ├── attention_model.py    # Attention architecture
    ├── rnn_trainer.py        # RNN training logic
    └── attention_trainer.py  # Attention training logic
```

## Installation

1. Navigate to the project directory:
   ```bash
   cd thesis_code/Chp3_EfficiencyMaps/efficiency_map_predictor
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

   Or if using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The main entry point provides a simple command-line interface:

#### RNN Model with Efficiency Data
```bash
python main.py --model rnn --data_type efficiency --data_path /path/to/your/data
```

#### Attention Model with Power Factor Data
```bash
python main.py --model attention --data_type powerfactor --data_path /path/to/your/data
```

#### Custom Configuration
```bash
python main.py --model rnn --data_type efficiency --config custom_config.yaml
```

### Available Options

- `--model`: Choose model architecture
  - `rnn`: Simple RNN with GRU layers
  - `attention`: Attention-based encoder-decoder RNN
- `--data_type`: Choose prediction task
  - `efficiency`: Predict motor efficiency maps
  - `powerfactor`: Predict power factor maps
- `--config`: Path to custom configuration file (optional)
- `--data_path`: Override data path from config (optional)

## Configuration Files

### RNN Configuration (`config_rnn.yaml`)

```yaml
# Model Architecture
model:
  architecture: "rnn"
  bidirectional: true
  gru_units: 128
  dense_units: 64
  input_dim: 14

# Training Configuration
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "nadam"

# Data Configuration
data:
  data_path: "path/to/your/data"
  data_type: "efficiency"
  normalization: "divide_by_100"

# Logging Configuration
logging:
  log_dir: "runs/rnn_training"
  save_model: true
  model_save_path: "models/rnn_model"
```

### Attention Configuration (`config_attention.yaml`)

```yaml
# Model Architecture
model:
  architecture: "attention"
  bidirectional: true
  gru_units: 128
  dense_units: 64
  dropout: 0.5
  input_dim: 14

# Training Configuration
training:
  epochs: 30
  batch_size: 32
  learning_rate: 0.001
  optimizer: "nadam"

# Data Configuration
data:
  data_path: "path/to/your/data"
  data_type: "efficiency"
  normalization: "divide_by_100"

# Logging Configuration
logging:
  log_dir: "runs/attention_training"
  save_model: true
  model_save_path: "models/attention_model"
```

## Data Requirements

### File Structure
The data should be organized as follows:
```
data_directory/
├── TrainText/           # Efficiency data text files
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
└── datasetEffMapText_xReal.h5  # Efficiency HDF5 data

# OR for power factor data:
data_directory/
├── TrainTextPf/         # Power factor data text files
│   ├── 1.txt
│   ├── 2.txt
│   └── ...
└── datasetPfMapText_xReal.h5   # Power factor HDF5 data
```

### Data Format
Each text file should contain:
- **First line**: Two integers - sequence length and input dimension
- **Remaining lines**: Operating data with efficiency/power factor as last column
- **Format**: `Torque Speed Current AdvanceAngle Nbase PowerFactor/Efficiency`

### HDF5 Format
The HDF5 file should contain:
- **Dataset**: `/DS_TrINP` with shape (num_files, 15, input_dim)
- **Structure**: Input features for each operating point

## Output

Each training run creates a timestamped results directory containing:

- `logs/`: Training logs and TensorBoard files
- `model/`: Saved trained model (if enabled)
- `training_YYYYMMDD_HHMMSS.log`: Detailed training log
- `tensorboard/`: TensorBoard event files

## Architecture Details

### RNN Model
- **Architecture**: Simple GRU-based RNN
- **Layers**:
  - Dense layers for input feature processing
  - Bidirectional GRU layers (optional)
  - TimeDistributed output layer
- **Use Case**: Baseline model for efficiency/power factor prediction

### Attention Model
- **Architecture**: Encoder-decoder with attention mechanism
- **Layers**:
  - Dense layers for input feature processing
  - Bidirectional GRU encoder and decoder
  - Attention mechanism with dot product
  - Batch normalization for stability
- **Use Case**: Advanced model for complex sequence modeling

## Model Comparison

| Feature | RNN Model | Attention Model |
|---------|-----------|-----------------|
| Complexity | Simple | Advanced |
| Attention | No | Yes |
| Parameters | Fewer | More |
| Training Time | Faster | Slower |
| Performance | Good | Better |

## Extending the Framework

### Adding New Models

1. Create new model class in `src/`
2. Create corresponding trainer class
3. Add configuration file
4. Update `main.py` to include new model

### Adding New Data Types

1. Update `DataLoader` class to handle new data type
2. Add data type to configuration files
3. Update command-line arguments in `main.py`

## Troubleshooting

### Common Issues

1. **Data Path Not Found**
   - Ensure data path exists and contains required files
   - Check file permissions
   - Update path in configuration file

2. **Import Errors**
   - Make sure you're running from project root
   - Check Python path includes src directory
   - Install missing dependencies

3. **Memory Issues**
   - Reduce batch size in configuration
   - Use smaller model configurations
   - Process data in smaller chunks

4. **Training Instability**
   - Adjust learning rate in configuration
   - Enable/disable bidirectional layers
   - Modify dropout rates

### Debug Mode

Enable debug logging by modifying the logging configuration:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{efficiency_map_predictor,
  title={Efficiency Map Predictor: RNN and Attention Architectures},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## License

This project is part of a thesis research. Please refer to your institution's guidelines for usage and attribution.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For questions or issues, please create an issue in the repository or contact the maintainers.
