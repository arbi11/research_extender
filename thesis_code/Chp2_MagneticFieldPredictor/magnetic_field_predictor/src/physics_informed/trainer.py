"""
Physics-Informed Neural Network trainer for magnetic field prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

from .xpinn_model import XPINN


class PINNTrainer:
    """Physics-Informed Neural Network trainer."""

    def __init__(self, config_path=None):
        """Initialize the trainer."""
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        import yaml

        config_file = "config_pinn.yaml" if not self.config_path else self.config_path
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'xpinn': {
                    'layers1': [2, 400, 400, 100, 1],
                    'layers2': [2, 100, 100, 40, 1],
                    'mu1': 1,
                    'mu2': 1,
                    'scaling_factor': 20,
                    'multiplier': 20
                },
                'training': {
                    'max_iter': 25000,
                    'adam_lr': 0.0008,
                    'n_f1': 1000,
                    'n_f2': 200,
                    'n_ub': 500,
                    'n_i1': 250
                },
                'physics': {
                    'equation': 'poisson',
                    'domain_decomposition': 'interface'
                }
            }

    def generate_training_data(self):
        """Generate training data for XPINN."""
        # Domain bounds
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])

        # Boundary points
        N_ub = self.config['training']['n_ub']
        X_ub = lb + (ub - lb) * np.random.rand(N_ub, 2)
        ub = np.zeros((N_ub, 1))

        # Collocation points for subdomain 1
        N_f1 = self.config['training']['n_f1']
        X_f1 = lb + (ub - lb) * np.random.rand(N_f1, 2)

        # Collocation points for subdomain 2
        N_f2 = self.config['training']['n_f2']
        X_f2 = lb + (ub - lb) * np.random.rand(N_f2, 2)

        # Interface points
        N_i1 = self.config['training']['n_i1']
        X_i1 = np.zeros((N_i1, 2))
        X_i1[:, 0] = np.random.rand(N_i1)
        X_i1[:, 1] = 0.0  # Interface at y=0

        # Interface solution (can be zero or some known function)
        u_i1 = np.zeros((N_i1, 1))

        return X_ub, ub, X_f1, X_f2, X_i1, u_i1

    def train(self, results_dir):
        """Main training function."""
        print("Setting up XPINN model...")

        # Get configuration
        layers1 = self.config['xpinn']['layers1']
        layers2 = self.config['xpinn']['layers2']
        mu1 = self.config['xpinn']['mu1']
        mu2 = self.config['xpinn']['mu2']

        # Generate training data
        X_ub, ub, X_f1, X_f2, X_i1, u_i1 = self.generate_training_data()

        # Create XPINN model
        model = XPINN(layers1, layers2, mu1, mu2)

        # Setup training
        model.setup_training(X_ub, ub, X_f1, X_f2, X_i1, u_i1)

        print("Starting training...")
        max_iter = self.config['training']['max_iter']

        # Training loop
        start_time = time.time()
        loss_history = []

        for it in range(max_iter):
            model.train_step()

            if it % 1000 == 0:
                loss1, loss2 = model.get_loss()
                total_loss = loss1 + loss2
                loss_history.append(total_loss)

                elapsed = time.time() - start_time
                print(f'Iteration: {it}, Loss1: {loss1:.3e}, Loss2: {loss2:.3e}, Total: {total_loss:.3e}, Time: {elapsed:.2f}s')

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        # Save model
        model_save_path = results_dir / "model"
        model_save_path.mkdir(exist_ok=True)

        # Save model weights (in a simple format)
        model_weights = {
            'weights1': [w.eval(session=model.sess).tolist() for w in model.weights1],
            'biases1': [b.eval(session=model.sess).tolist() for b in model.biases1],
            'A1': [a.eval(session=model.sess) for a in model.A1],
            'weights2': [w.eval(session=model.sess).tolist() for w in model.weights2],
            'biases2': [b.eval(session=model.sess).tolist() for b in model.biases2],
            'A2': [a.eval(session=model.sess) for a in model.A2]
        }

        with open(model_save_path / "xpinn_weights.json", 'w') as f:
            json.dump(model_weights, f)

        # Save training metrics
        metrics = {
            'max_iter': max_iter,
            'adam_lr': self.config['training']['adam_lr'],
            'n_f1': self.config['training']['n_f1'],
            'n_f2': self.config['training']['n_f2'],
            'n_ub': self.config['training']['n_ub'],
            'n_i1': self.config['training']['n_i1'],
            'total_time': total_time,
            'final_loss1': loss1,
            'final_loss2': loss2,
            'final_total_loss': loss1 + loss2,
            'loss_history': loss_history
        }

        with open(results_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Generate test predictions
        print("Generating test predictions...")
        N_test = 100
        X_test1 = np.random.rand(N_test, 2) * 2 - 1  # [-1, 1] domain
        X_test2 = np.random.rand(N_test, 2) * 2 - 1

        u_pred1, u_pred2 = model.predict(X_test1, X_test2)

        # Save predictions
        test_results = {
            'X_test1': X_test1.tolist(),
            'X_test2': X_test2.tolist(),
            'u_pred1': u_pred1.tolist(),
            'u_pred2': u_pred2.tolist()
        }

        with open(results_dir / "test_predictions.json", 'w') as f:
            json.dump(test_results, f, indent=2)

        print("PINN training completed!")
        print(f"Results saved to: {results_dir}")
