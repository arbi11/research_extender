"""
Deep Learning trainer for magnetic field prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import glob
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from .model import get_model


class DLTrainer:
    """Deep Learning trainer for magnetic field prediction."""

    def __init__(self, config_path=None):
        """Initialize the trainer."""
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        """Load configuration from YAML file."""
        import yaml

        config_file = "config_dl.yaml" if not self.config_path else self.config_path
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'model': {'img_size': [400, 400], 'channels': 1},
                'training': {'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001},
                'data': {'normalization': 'minmax'}
            }

    def get_B_stats(self, files, folder_path):
        """Get B-field statistics from data files."""
        B_stats = {'max': 0, 'min': 1}

        for file in files:
            file_path = folder_path / file
            df = pd.read_csv(file_path, header=None)
            df.rename(columns={0: 'x', 1: 'y', 2: 'B', 3: 'mat'}, inplace=True)

            if df['B'].min() < B_stats['min']:
                B_stats['min'] = df['B'].min()

            if df['B'].max() > B_stats['max']:
                B_stats['max'] = df['B'].max()

        return B_stats['min'], B_stats['max']

    def generate_geom_array(self, ds, cr, cm):
        """Generate geometry array for given parameters."""
        x = np.arange(0, 400)
        y = np.arange(0, 400)
        arr = np.zeros((y.size, x.size))

        cx = x.size/2
        cy = y.size/2
        r = (cr * 200) / ds

        mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
        arr[mask] = cm

        return arr

    def data_feed(self, index, files, folder_path, batch_size):
        """Generate data batch for training."""
        if (index + batch_size) >= len(files):
            index = len(files) - batch_size - 1

        batch_files = files[index : index + batch_size]

        X = np.zeros([batch_size, 400, 400, 1])
        y = np.zeros([batch_size, 400, 400, 1])

        for inx, csv_file in enumerate(batch_files):
            file_path = folder_path / csv_file

            coil_radius = float(csv_file[3: csv_file.find('_ds_')])
            domain_size = float(csv_file[csv_file.find('_ds_') + 4: csv_file.find('_cm_')])
            current_magnitude = float(csv_file[csv_file.find('_cm_') + 4 : -4])

            df = pd.read_csv(file_path, header=None)
            df.rename(columns={0: 'x', 1: 'y', 2: 'B', 3: 'mat'}, inplace=True)

            # Generate geometry
            arr = self.generate_geom_array(domain_size, coil_radius, current_magnitude)

            # Generate B field plot
            points = df[['x', 'y']].to_numpy()
            values = df[['B']].to_numpy().squeeze()

            grid_x, grid_y = np.mgrid[-domain_size:domain_size:400j, -domain_size:domain_size:400j]
            grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')

            X[inx, :, :, 0] = arr
            y[inx, :, :, 0] = grid_z1.T

        y_norm = y * 1000
        return X, y_norm, y

    def my_training(self, batch_size, epochs, no_iters, lr):
        """Custom training loop."""
        # Setup data
        main_folder_path = Path('.')
        folder_path = main_folder_path / 'data' / 'raw' / 'NL_Data2'
        files = glob.glob1(folder_path, "*.csv")
        self.data_size = len(files)

        B_min, B_max = self.get_B_stats(files, folder_path)
        self.norm_denom = B_max - B_min

        # Create model
        model = get_model(img_size=(400, 400))

        # Setup optimizer
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()

            for i in range(no_iters):
                index = np.random.randint(self.data_size - batch_size)
                X, Y, _ = self.data_feed(index, files, folder_path, batch_size)

                with tf.GradientTape() as tape:
                    y_pred = model(X, training=True)
                    loss_ = tf.keras.losses.mean_squared_error(Y, y_pred)

                gradients = tape.gradient(loss_, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            end_time = time.time()
            print(f'Epoch: {epoch+1} | Time: {end_time - start_time}')

        return model

    def train(self, results_dir):
        """Main training function."""
        print("Setting up data...")
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        lr = self.config['training']['learning_rate']

        no_iters = int(2 * (self.data_size // batch_size))

        print("Starting training...")
        model = self.my_training(batch_size, epochs, no_iters, lr)

        # Save model
        model_save_path = results_dir / "model"
        model_save_path.mkdir(exist_ok=True)
        model.save(str(model_save_path / "field_predictor_model.h5"))

        print(f"Model saved to: {model_save_path}")

        # Save training metrics
        metrics = {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'data_size': self.data_size,
            'final_loss': 'N/A'  # Could be calculated if needed
        }

        import json
        with open(results_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        print("Training completed!")
