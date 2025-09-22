#!/usr/bin/env python3
"""
C-core Reinforcement Learning Optimization - Main Entry Point

This script provides a unified interface for running different reinforcement
learning algorithms (A2C, PPO, DQN) for C-core electromagnetic actuator optimization.

Usage:
    python main.py --method a2c          # Run A2C algorithm
    python main.py --method ppo          # Run PPO algorithm  
    python main.py --method dqn          # Run DQN algorithm
"""

import argparse
import sys
import os
import logging
import yaml
from datetime import datetime
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_a2c(config: dict, results_dir: Path):
    """Run the A2C algorithm"""
    print("🚀 Starting A2C (Advantage Actor-Critic) training...")
    
    from src.a2c.trainer import A2CTrainer
    
    trainer = A2CTrainer(config, results_dir)
    results = trainer.train()
    trainer.close()
    
    print("✅ A2C training completed successfully!")
    print(f"📊 Final performance:")
    print(f"   - Mean reward (last 10): {results['final_performance']['mean_reward_last_10']:.2f}")
    print(f"   - Best reward: {results['final_performance']['best_reward']:.2f}")
    print(f"   - Total episodes: {results['total_episodes']}")
    print(f"   - Training time: {results['training_time']:.2f}s")


def run_ppo(config: dict, results_dir: Path):
    """Run the PPO algorithm"""
    print("🚀 Starting PPO (Proximal Policy Optimization) training...")
    
    # TODO: Implement PPO trainer
    print("❌ PPO implementation not yet available")
    print("   Please use A2C for now or implement PPO trainer")


def run_dqn(config: dict, results_dir: Path):
    """Run the DQN algorithm"""
    print("🚀 Starting DQN (Deep Q-Network) training...")
    
    # TODO: Implement DQN trainer
    print("❌ DQN implementation not yet available")
    print("   Please use A2C for now or implement DQN trainer")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="C-core RL Optimization - Multiple RL algorithms for electromagnetic design"
    )
    
    parser.add_argument(
        '--method',
        choices=['a2c', 'ppo', 'dqn'],
        required=True,
        help='RL method to run: a2c (Advantage Actor-Critic), ppo (Proximal Policy Optimization), or dqn (Deep Q-Network)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (optional)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory to save results (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = f"config_{args.method}.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print(f"   Available configs: config_a2c.yaml, config_ppo.yaml, config_dqn.yaml")
        sys.exit(1)
    
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Create results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%b%d_%H%M")
        results_dir = Path("results") / f"{timestamp}_{args.method.upper()}"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.get('logging', {}).get('log_level', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(results_dir / "main.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print(f"🎯 C-core RL Optimization Framework")
    print(f"📋 Method: {args.method.upper()}")
    print(f"⚙️  Config: {config_path}")
    print(f"📁 Results: {results_dir}")
    print("=" * 60)
    
    logger.info(f"Starting {args.method.upper()} training")
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Validate FEMM path
    femm_path = config.get('paths', {}).get('femm_path', 'C:\\femm42')
    if not os.path.exists(femm_path):
        print(f"⚠️  Warning: FEMM path not found: {femm_path}")
        print(f"   Please ensure FEMM is installed and update the config file")
        print(f"   Training will continue but may fail during environment initialization")
    
    # Run the selected method
    try:
        if args.method == 'a2c':
            run_a2c(config, results_dir)
        elif args.method == 'ppo':
            run_ppo(config, results_dir)
        elif args.method == 'dqn':
            run_dqn(config, results_dir)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        logger.info("Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Final status
    print("=" * 60)
    print("🎉 Training completed successfully!")
    print(f"📊 Check results in: {results_dir}")
    print(f"📈 Training logs: {results_dir}/training.log")
    print(f"💾 Model checkpoints: {results_dir}/checkpoints/")
    print(f"🎬 Renderings: {results_dir}/renders/")
    print("=" * 60)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
