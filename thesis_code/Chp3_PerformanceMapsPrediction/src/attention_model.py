"""
Attention-based RNN Model for Efficiency Map Prediction

This module contains the attention-based RNN model architecture for predicting efficiency maps.
Based on the original rnn_tf2_attention_eff.py implementation with encoder-decoder attention.
"""

import tensorflow as tf
from tensorflow.keras.layers import concatenate


class AttentionModel(tf.keras.Model):
    """
    Attention-based RNN Model for efficiency/power factor prediction.

    This model uses an encoder-decoder architecture with attention mechanism
    to focus on relevant parts of the input sequence.

    Args:
        bi (bool): Whether to use bidirectional RNN layers
        input_dim (int): Input dimension for the dense layers
        gru_units (int): Number of GRU units
        dense_units (int): Number of dense units
        dropout (float): Dropout rate for regularization
    """

    def __init__(self, bi=False, input_dim=14, gru_units=128, dense_units=64, dropout=0.5):
        super(AttentionModel, self).__init__()

        self.bi = bi
        self.input_dim = input_dim
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout = dropout

        # Initial dense layers for processing input features
        self.hidden1 = tf.keras.layers.Dense(units=self.dense_units)
        self.out = tf.keras.layers.Dense(units=self.gru_units, activation='tanh')

        # Encoder GRU layers
        if self.bi:
            self.gru_enc_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True, return_state=True))

            self.gru_enc_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True))

            self.gru_dec_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True))
        else:
            self.gru_enc_1 = tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True, return_state=True)

            self.gru_enc_2 = tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True)

            self.gru_dec_1 = tf.keras.layers.GRU(
                self.gru_units, activation='elu',
                return_sequences=True)

        # Batch normalization layers
        if self.bi:
            self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.6)
            self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.6)
        else:
            self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.6)

        # Attention mechanism
        self.attention1 = tf.keras.layers.Dot(axes=(2, 2))
        self.attention2 = tf.keras.layers.Activation('softmax')

        # Context and output layers
        self.bn_con = tf.keras.layers.BatchNormalization(momentum=0.6)
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.6)

        self.context1 = tf.keras.layers.Dot(axes=(2, 1))
        self.context2 = tf.keras.layers.BatchNormalization(momentum=0.6)

        # Output layer
        self.dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(1, activation="sigmoid"))

    def call(self, inputs, training=False):
        """
        Forward pass through the attention model.

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

        # Encoder
        if self.bi:
            initial_state = [h, h]
            encoder_stack_h, encoder_last_h1, encoder_last_h2 = self.gru_enc_1(
                input2, initial_state=initial_state, mask=mask)
        else:
            initial_state = h
            encoder_stack_h, encoder_last_h = self.gru_enc_1(
                input2, initial_state=initial_state, mask=mask)

        # Batch normalization on encoder output
        if self.bi:
            encoder_last_h1 = self.bn1(encoder_last_h1)
            encoder_last_h2 = self.bn2(encoder_last_h2)
            encoder_last_h = concatenate([encoder_last_h1, encoder_last_h2])
        else:
            encoder_last_h = self.bn1(encoder_last_h)

        # Prepare decoder input
        decoder_input = tf.expand_dims(encoder_last_h, axis=1)
        decoder_input = tf.repeat(decoder_input, input2.shape[1], axis=1)

        # Decoder initial state
        if self.bi:
            dec_initial_state = [encoder_last_h1, encoder_last_h2]
        else:
            dec_initial_state = encoder_last_h

        # Decoder
        decoder_stack_h = self.gru_dec_1(decoder_input,
                                       initial_state=dec_initial_state,
                                       mask=mask)

        # Attention mechanism
        attention = self.attention1([decoder_stack_h, encoder_stack_h])
        attention = self.attention2(attention)

        # Context vector
        context = self.context1([attention, encoder_stack_h])
        context = self.bn_con(context)

        # Combine context and decoder output
        decoder_combined_context = concatenate([context, decoder_stack_h])

        # Output
        output = self.dense(decoder_combined_context, mask=mask)

        return output

    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'bi': self.bi,
            'input_dim': self.input_dim,
            'gru_units': self.gru_units,
            'dense_units': self.dense_units,
            'dropout': self.dropout
        }

    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        return cls(**config)
