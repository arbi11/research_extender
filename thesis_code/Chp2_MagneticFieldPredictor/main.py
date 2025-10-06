#!/usr/bin/env python3
"""
Magnetic Field Predictor - Main Entry Point

This script provides a unified interface for running both Deep Learning
and Physics-Informed Neural Network approaches for magnetic field prediction.

Usage:
    python main.py --method dl          # Run Deep Learning approach
    python main.py --method pinn        # Run Physics-Informed Neural Network approach
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def run_deep_learning(results_dir):
    """Run the Deep Learning approach."""
    print("🚀 Starting Deep Learning approach...")

    from src.deep_learning.trainer import DLTrainer

    trainer = DLTrainer()
    trainer.train(results_dir)

    print("✅ Deep Learning training completed successfully!")
    print(f"📁 Results saved to: {results_dir}")


def run_physics_informed(results_dir):
    """Run the Physics-Informed Neural Network approach."""
    print("🚀 Starting Physics-Informed Neural Network approach...")

    from src.physics_informed.trainer import PINNTrainer

    trainer = PINNTrainer()
    trainer.train(results_dir)

    print("✅ PINN training completed successfully!")
    print(f"📁 Results saved to: {results_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Magnetic Field Predictor - DL and PINN approaches"
    )

    parser.add_argument(
        '--method',
        choices=['dl', 'pinn'],
        required=True,
        help='Method to run: dl (Deep Learning) or pinn (Physics-Informed NN)'
    )

    args = parser.parse_args()

    # Create results directory
    timestamp = datetime.now().strftime("%Y_%b%d_%H%M")
    results_dir = Path("results") / f"{timestamp}_{args.method.upper()}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dir / "training.log"),
            logging.StreamHandler()
        ]
    )

    print(f"📁 Results will be saved to: {results_dir}")
    print("-" * 60)

    # Run the selected method
    if args.method == 'dl':
        run_deep_learning(results_dir)
    elif args.method == 'pinn':
        run_physics_informed(results_dir)

    # Final status
    print("-" * 60)
    print("🎉 Training completed successfully!")
    print(f"📊 Check results in: {results_dir}")


if __name__ == "__main__":
    main()
