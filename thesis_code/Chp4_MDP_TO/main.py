#!/usr/bin/env python3
"""
MDP Topology Optimizer - Main Entry Point

A unified framework for topology optimization of electromagnetic devices using
reinforcement learning (Q-Learning) and evolutionary algorithms (Genetic Algorithm).

This system supports two environments:
1. SynRM (Synchronous Reluctance Motor) - Rotor geometry optimization
2. C-core - C-shaped magnetic core optimization

Usage:
    python main.py --method ql --environment synrm    # Q-learning on SynRM
    python main.py --method ga --environment ccore    # GA on C-core
    python main.py --method compare                    # Compare both methods
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
import yaml

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from q_learning.trainer import QLearningTrainer
from genetic_algorithm.algorithm import GeneticAlgorithm
from environments.synrm_env import SynRMEnvironment
from environments.ccore_env import CCoreEnvironment
from utils.visualization import TopologyVisualizer


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_results_directory(base_dir="results"):
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(base_dir) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def setup_logging(results_dir, log_level="INFO"):
    """Setup logging configuration."""
    log_file = results_dir / "training.log"

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def run_q_learning(environment_name, config, results_dir):
    """Run Q-Learning optimization."""
    print("🚀 Starting Q-Learning optimization...")

    if environment_name.lower() == "synrm":
        env = SynRMEnvironment(config)
    elif environment_name.lower() == "ccore":
        env = CCoreEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {environment_name}")

    trainer = QLearningTrainer(env, config)
    trainer.train(results_dir)

    print("✅ Q-Learning training completed successfully!")
    print(f"📁 Results saved to: {results_dir}")


def run_genetic_algorithm(environment_name, config, results_dir):
    """Run Genetic Algorithm optimization."""
    print("🚀 Starting Genetic Algorithm optimization...")

    if environment_name.lower() == "synrm":
        env = SynRMEnvironment(config)
    elif environment_name.lower() == "ccore":
        env = CCoreEnvironment(config)
    else:
        raise ValueError(f"Unknown environment: {environment_name}")

    ga = GeneticAlgorithm(env, config)
    ga.optimize(results_dir)

    print("✅ Genetic Algorithm optimization completed successfully!")
    print(f"📁 Results saved to: {results_dir}")


def run_comparison(config, results_dir):
    """Run comparison between Q-Learning and Genetic Algorithm."""
    print("🚀 Starting method comparison...")

    for env_name in ["synrm", "ccore"]:
        results[env_name] = {}

        for method_name in ["ql", "ga"]:
            print(f"Running {method_name.upper()} on {env_name.upper()}...")

            if method_name == "ql":
                run_q_learning(env_name, config, results_dir / f"{env_name}_{method_name}")
            else:
                run_genetic_algorithm(env_name, config, results_dir / f"{env_name}_{method_name}")

    print("✅ Comparison completed successfully!")
    print(f"📁 Results saved to: {results_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MDP Topology Optimizer - RL and GA approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --method ql --environment synrm    # Q-learning on SynRM
  python main.py --method ga --environment ccore    # GA on C-core
  python main.py --method compare                    # Compare both methods
        """
    )

    parser.add_argument(
        '--method',
        choices=['ql', 'ga', 'compare'],
        required=True,
        help='Optimization method: ql (Q-Learning), ga (Genetic Algorithm), compare (both)'
    )

    parser.add_argument(
        '--environment',
        choices=['synrm', 'ccore'],
        help='Environment: synrm (Synchronous Reluctance Motor), ccore (C-core)'
    )

    args = parser.parse_args()

    # Load configuration
    print("📋 Loading configuration...")
    config = load_config()
    print("✅ Configuration loaded successfully!")

    # Create results directory
    results_dir = create_results_directory()
    print(f"📁 Results will be saved to: {results_dir}")

    # Setup logging
    setup_logging(results_dir)

    logging.info("MDP Topology Optimizer Starting")
    logging.info(f"Method: {args.method}")
    logging.info(f"Environment: {args.environment}")

    print("-" * 60)

    # Run the selected method
    if args.method == 'ql':
        if not args.environment:
            parser.error("--environment is required when using --method ql")
        run_q_learning(args.environment, config, results_dir)

    elif args.method == 'ga':
        if not args.environment:
            parser.error("--environment is required when using --method ga")
        run_genetic_algorithm(args.environment, config, results_dir)

    elif args.method == 'compare':
        run_comparison(config, results_dir)

    # Final status
    print("-" * 60)
    print("🎉 Optimization completed successfully!")
    print(f"📊 Check results in: {results_dir}")


if __name__ == "__main__":
    main()
