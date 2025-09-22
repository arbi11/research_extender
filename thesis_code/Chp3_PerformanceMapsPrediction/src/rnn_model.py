"""
RNN Model for Efficiency Map Prediction

This module contains the RNN model architecture for predicting efficiency maps.
Based on the original rnn_tf2_fnn_eff.py implementation.
"""

import tensorflow as tf


class RNNModel(tf.keras.Model):
    """
    RNN Model for efficiency/power factor prediction.

    Args:
        bi (bool): Whether to use bidirectional RNN layers
        input_dim (int): Input dimension for the dense layers
        gru_units (int): Number of GRU units
        dense_units (int): Number of dense units
    """

    def __init__(self, bi=False, input_dim=14, gru_units=128, dense_units=64):
        super(RNNModel, self).__init__()

        self.bi = bi
        self.input_dim = input_dim
        self.gru_units = gru_units
        self.dense_units = dense_units

        # Initial dense layers for processing input features
        self.hidden1 = tf.keras.layers.Dense(units=self.dense_units)
        self.out = tf.keras.layers.Dense(units=self.gru_units, activation='tanh')

        # GRU layers
        if self.bi:
            self.gru_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True, return_state=True))

            self.gru_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                int(self.gru_units / 2), activation='elu',
                return_sequences=True))
        else:
            self.gru_1 = tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True, return_state=True)

            self.gru_2 = tf.keras.layers.GRU(
                int(self.gru_units / 2), activation='elu',
                return_sequences=True)

        # Output layer
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation="sigmoid"))

    def call(self, inputs, training=False):
        """
        Forward pass through the model.

        Args:
            inputs: List containing [X_inp, X, mask]
            training: Whether in training mode

        Returns:
            Model predictions
        """
        input1, input2, mask = inputs

        # Process input features
        h = self.hidden1(input1)
        h = self.out(h)

        # RNN layers
        if self.bi:
            initial_state = [h, h]
            x, state_h, state_c = self.gru_1(input2,
                                           initial_state=initial_state,
                                           mask=mask)
        else:
            initial_state = h
            x, state_h = self.gru_1(input2,
                                  initial_state=initial_state,
                                  mask=mask)

        x = self.gru_2(x, mask=mask)
        output = self.dense(x, mask=mask)

        return output

    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'bi': self.bi,
            'input_dim': self.input_dim,
            'gru_units': self.gru_units,
            'dense_units': self.dense_units
        }

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
