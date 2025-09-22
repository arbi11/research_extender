"""
RNN Model Trainer for Efficiency Map Prediction

This module contains the training logic for RNN models.
Based on the original training code from rnn_tf2_fnn_eff.py.
"""

import time
import numpy as np
import tensorflow as tf
from .rnn_model import RNNModel


class RNNTrainer:
    """
    Trainer class for RNN models.

    Args:
        model (RNNModel): The RNN model to train
        data_loader (DataLoader): Data loader for training data
        config (dict): Training configuration
    """

    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config

        # Extract configuration
        self.epochs = config.get('epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)

        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.learning_rate)

        # Setup loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Setup metrics
        self.mean_loss = tf.keras.metrics.Mean()

        # Setup TensorBoard logging
        self.log_dir = config.get('log_dir', 'runs/rnn_training')
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def epoch_time(self, start_time, end_time):
        """Calculate elapsed time for epoch."""
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train_step(self, X_inp, X, Y, mask):
        """Single training step."""
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.model([X_inp, X, mask], training=True)

            # Calculate loss
            loss = self.loss_fn(Y, y_pred)

        # Calculate gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update metrics
        self.mean_loss(loss)

        return loss, y_pred

    def train_epoch(self, epoch):
        """Train for one epoch."""
        start_time = time.time()
        epoch_losses = []

        # Calculate number of iterations per epoch
        num_batches = len(self.data_loader.names)
        num_iters = int(num_batches / self.batch_size) * 2

        for i in range(num_iters):
            # Get random batch
            index = np.random.randint(num_batches - self.batch_size)
            X_inp, X, Y, src_len = self.data_loader.data_feed(index)

            # Normalize targets to 0-1 range
            Y = Y / 100.0
            Y = np.expand_dims(Y, axis=-1)

            # Create mask for variable-length sequences
            mask = np.logical_not(X)  # Find zero values
            mask = np.logical_not(np.all(mask, axis=-1))

            # Training step
            loss, y_pred = self.train_step(X_inp, X, Y, mask)
            epoch_losses.append(loss.numpy())

        end_time = time.time()
        epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

        # Calculate epoch statistics
        avg_loss = np.mean(epoch_losses)

        print(f'Epoch: {epoch+1:2d} | Time: {epoch_mins}m {epoch_secs}s | '
              f'Avg Loss: {avg_loss:.6f}')

        return avg_loss

    def train(self):
        """Main training loop."""
        print("🚀 Starting RNN model training...")
        print(f"Training for {self.epochs} epochs")
        print("-" * 60)

        # Training loop
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(epoch)

            # Log to TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)

                # Log model weights
                try:
                    for layer in self.model.layers:
                        layer_weights = layer.get_weights()
                        layer_name = layer.name
                        for idx, layer_weights_array in enumerate(layer_weights):
                            tf.summary.histogram(f'{layer_name}_{idx}',
                                               layer_weights_array, epoch)
                except Exception as e:
                    print(f"Warning: Could not log weights: {e}")

            # Qualitative evaluation (visualization)
            try:
                # Get a sample batch for visualization
                index = np.random.randint(len(self.data_loader.names) - self.batch_size)
                X_inp, X, Y, src_len = self.data_loader.data_feed(index)
                Y = Y / 100.0
                Y = np.expand_dims(Y, axis=-1)
                mask = np.logical_not(X)
                mask = np.logical_not(np.all(mask, axis=-1))

                # Get predictions
                y_pred = self.model([X_inp, X, mask], training=False)
                y_pred = y_pred.numpy()

                # Generate visualization
                self.data_loader.qualitative_measure(
                    X[0, :, 1], X[0, :, 0], Y[0, :, 0], y_pred[0, :, 0], index)

            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")

        # Flush TensorBoard logs
        tf.summary.flush(self.summary_writer)

        print("-" * 60)
        print("✅ RNN training completed successfully!")
        print(f"📊 Check TensorBoard logs at: {self.log_dir}")

        return self.model

    def evaluate(self, test_data_path=None):
        """Evaluate the trained model."""
        print("🔍 Evaluating RNN model...")

        if test_data_path:
            # Load test data
            test_loader = self.data_loader.__class__(
                test_data_path, self.data_loader.data_type, self.batch_size)
        else:
            test_loader = self.data_loader

        # Calculate metrics
        total_loss = 0
        num_batches = 0

        for i in range(min(10, len(test_loader.names) - self.batch_size)):  # Test on 10 batches
            index = i * self.batch_size
            X_inp, X, Y, src_len = test_loader.data_feed(index)

            Y = Y / 100.0
            Y = np.expand_dims(Y, axis=-1)

            mask = np.logical_not(X)
            mask = np.logical_not(np.all(mask, axis=-1))

            y_pred = self.model([X_inp, X, mask], training=False)
            loss = self.loss_fn(Y, y_pred)

            total_loss += loss.numpy()
            num_batches += 1

        avg_test_loss = total_loss / num_batches
        print(f"📈 Average Test Loss: {avg_test_loss".6f"}")

        return avg_test_loss

    def save_model(self, filepath):
        """Save the trained model."""
        self.model.save(filepath)
        print(f"💾 Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath, custom_objects={'RNNModel': RNNModel})
        print(f"📂 Model loaded from: {filepath}")
        return self.model
