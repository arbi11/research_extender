#!/usr/bin/env python3
"""
Efficiency Map Predictor - Main Entry Point

This script provides a unified interface for running different RNN architectures
for efficiency and power factor prediction.

Usage:
    python main.py --model rnn --data_type efficiency
    python main.py --model attention --data_type powerfactor
    python main.py --model rnn --data_type efficiency --config custom_config.yaml
"""

import argparse
import sys
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.data_loader import DataLoader
from src.rnn_model import RNNModel
from src.attention_model import AttentionModel
from src.rnn_trainer import RNNTrainer
from src.attention_trainer import AttentionTrainer


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def setup_logging(log_dir):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file


def create_results_directory(model_type, data_type):
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"{timestamp}_{model_type}_{data_type}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_rnn_model(config, data_path, data_type):
    """Run the RNN model training."""
    print("🚀 Starting RNN model training...")

    # Create results directory
    results_dir = create_results_directory("rnn", data_type)

    # Setup logging
    log_file = setup_logging(results_dir / "logs")
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"📋 Logs will be saved to: {log_file}")

    # Update config with results directory
    config['training']['log_dir'] = str(results_dir / "tensorboard")
    config['logging']['log_dir'] = str(results_dir / "logs")

    # Initialize data loader
    print(f"📊 Loading {data_type} data from: {data_path}")
    data_loader = DataLoader(data_path, data_type, config['training']['batch_size'])

    # Display data info
    data_info = data_loader.get_data_info()
    print(f"📊 Data loaded: {data_info['num_files']} files, "
          f"Input dim: {data_info['input_dimension']}, "
          f"Data type: {data_info['data_type']}")

    # Initialize model
    model_config = config['model']
    model = RNNModel(
        bi=model_config['bidirectional'],
        input_dim=model_config['input_dim'],
        gru_units=model_config['gru_units'],
        dense_units=model_config['dense_units']
    )

    # Initialize trainer
    trainer = RNNTrainer(model, data_loader, config['training'])

    # Train model
    trained_model = trainer.train()

    # Evaluate model
    test_loss = trainer.evaluate()
    print(f"📈 Final Test Loss: {test_loss:.6f}")

    # Save model if configured
    if config['logging'].get('save_model', True):
        model_path = results_dir / "model"
        trainer.save_model(str(model_path))
        print(f"💾 Model saved to: {model_path}")

    print("✅ RNN training completed successfully!")
    print(f"📊 Check results in: {results_dir}")

    return trained_model, results_dir


def run_attention_model(config, data_path, data_type):
    """Run the Attention model training."""
    print("🚀 Starting Attention model training...")

    # Create results directory
    results_dir = create_results_directory("attention", data_type)

    # Setup logging
    log_file = setup_logging(results_dir / "logs")
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"📋 Logs will be saved to: {log_file}")

    # Update config with results directory
    config['training']['log_dir'] = str(results_dir / "tensorboard")
    config['logging']['log_dir'] = str(results_dir / "logs")

    # Initialize data loader
    print(f"📊 Loading {data_type} data from: {data_path}")
    data_loader = DataLoader(data_path, data_type, config['training']['batch_size'])

    # Display data info
    data_info = data_loader.get_data_info()
    print(f"📊 Data loaded: {data_info['num_files']} files, "
          f"Input dim: {data_info['input_dimension']}, "
          f"Data type: {data_info['data_type']}")

    # Initialize model
    model_config = config['model']
    model = AttentionModel(
        bi=model_config['bidirectional'],
        input_dim=model_config['input_dim'],
        gru_units=model_config['gru_units'],
        dense_units=model_config['dense_units'],
        dropout=model_config['dropout']
    )

    # Initialize trainer
    trainer = AttentionTrainer(model, data_loader, config['training'])

    # Train model
    trained_model = trainer.train()

    # Evaluate model
    test_loss = trainer.evaluate()
    print(f"📈 Final Test Loss: {test_loss:.6f}")

    # Save model if configured
    if config['logging'].get('save_model', True):
        model_path = results_dir / "model"
        trainer.save_model(str(model_path))
        print(f"💾 Model saved to: {model_path}")

    print("✅ Attention model training completed successfully!")
    print(f"📊 Check results in: {results_dir}")

    return trained_model, results_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Efficiency Map Predictor - RNN and Attention architectures"
    )

    parser.add_argument(
        '--model',
        choices=['rnn', 'attention'],
        required=True,
        help='Model architecture: rnn (simple RNN) or attention (attention-based RNN)'
    )

    parser.add_argument(
        '--data_type',
        choices=['efficiency', 'powerfactor'],
        required=True,
        help='Data type: efficiency (efficiency maps) or powerfactor (power factor maps)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (optional)'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to data directory (overrides config file setting)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = f"config_{args.model}.yaml"

    if not Path(config_path).exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Make sure you're running from the project root directory")
        return 1

    config = load_config(config_path)
    print(f"⚙️  Loaded configuration from: {config_path}")

    # Override data path if provided
    if args.data_path:
        config['data']['data_path'] = args.data_path

    data_path = config['data']['data_path']

    # Validate data path
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        print("💡 Please update the data_path in your configuration file or use --data_path")
        return 1

    # Update config with command line arguments
    config['data']['data_type'] = args.data_type

    print("-" * 60)
    print(f"🤖 Model: {args.model.upper()}")
    print(f"📊 Data Type: {args.data_type}")
    print(f"📁 Data Path: {data_path}")
    print("-" * 60)

    # Run the selected model
    try:
        if args.model == 'rnn':
            model, results_dir = run_rnn_model(config, data_path, args.data_type)
        elif args.model == 'attention':
            model, results_dir = run_attention_model(config, data_path, args.data_type)

        print("-" * 60)
        print("🎉 Training completed successfully!")
        print(f"📊 Check results in: {results_dir}")

        return 0

    except Exception as e:
        print(f"❌ Error during training: {e}")
        logging.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
