"""
Extended Physics-Informed Neural Network (XPINN) model for magnetic field prediction.
"""

import numpy as np
import tensorflow as tf
import scipy.io


class XPINN:
    """Extended Physics-Informed Neural Network implementation."""

    def __init__(self, layers1, layers2, mu1=1, mu2=1):
        """Initialize the XPINN model."""
        self.layers1 = layers1
        self.layers2 = layers2
        self.mu1 = mu1
        self.mu2 = mu2

        self.multiplier = 20
        self.scaling_factor = 20

        # Initialize neural networks
        self.weights1, self.biases1, self.A1 = self.initialize_NN(layers1)
        self.weights2, self.biases2, self.A2 = self.initialize_NN(layers2)

        # TensorFlow session
        self.sess = tf.Session()

    def initialize_NN(self, layers):
        """Initialize neural network weights and biases."""
        weights = []
        biases = []
        A = []

        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.05, dtype=tf.float64)
            weights.append(W)
            biases.append(b)
            A.append(a)

        return weights, biases, A

    def xavier_init(self, size):
        """Xavier initialization for weights."""
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.to_double(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)), dtype=tf.float64)

    def neural_net_tanh(self, X, weights, biases, A):
        """Neural network with tanh activation."""
        num_layers = len(weights) + 1

        H = X
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(self.scaling_factor*A[l]*tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u1(self, x, y):
        """First sub-network for subdomain 1."""
        u = self.neural_net_tanh(tf.concat([x,y],1), self.weights1, self.biases1, self.A1)
        return u

    def net_u2(self, x, y):
        """Second sub-network for subdomain 2."""
        u = self.neural_net_tanh(tf.concat([x,y],1), self.weights2, self.biases2, self.A2)
        return u

    def net_f(self, x1, y1, x2, y2, xi1, yi1):
        """Compute physics-informed loss terms."""
        # Sub-Net1
        u1 = self.net_u1(x1,y1)
        u1_x = tf.gradients(u1, x1)[0]
        u1_y = tf.gradients(u1, y1)[0]

        d2 = tf.math.add(tf.math.square(u1_x), tf.math.square(u1_y))
        d2 = tf.math.scalar_mul(0.05, d2)
        d2 = tf.math.add(d2, 1)
        d1 = tf.math.divide(5000, d2)
        d1 = tf.math.add(d1, 200)
        c = tf.math.divide(1, d2)

        u1_x = tf.math.multiply(c, u1_x)
        u1_y = tf.math.multiply(c, u1_y)

        u1_xx = tf.gradients(u1_x, x1)[0]
        u1_yy = tf.gradients(u1_y, y1)[0]

        # Sub-Net2
        u2 = self.net_u2(x2,y2)
        u2_x = tf.gradients(u2, x2)[0]
        u2_y = tf.gradients(u2, y2)[0]

        u2_xx = tf.gradients(u2_x, x2)[0]
        u2_yy = tf.gradients(u2_y, y2)[0]

        # Sub-Net1, Interface 1
        u1i1 = self.net_u1(xi1,yi1)
        u1i1_x = tf.gradients(u1i1, xi1)[0]
        u1i1_y = tf.gradients(u1i1, yi1)[0]

        d2 = tf.math.add(tf.math.square(u1i1_x), tf.math.square(u1i1_y))
        d2 = tf.math.scalar_mul(0.05, d2)
        d2 = tf.math.add(d2, 1)
        d1 = tf.math.divide(5000, d2)
        d1 = tf.math.add(d1, 200)
        c = tf.math.divide(1, d2)

        u1i1_x = tf.math.multiply(c, u1i1_x)
        u1i1_y = tf.math.multiply(c, u1i1_y)

        u1i1_xx = tf.gradients(u1i1_x, xi1)[0]
        u1i1_yy = tf.gradients(u1i1_y, yi1)[0]

        # Sub-Net2, Interface 1
        u2i1 = self.net_u2(xi1,yi1)
        u2i1_x = tf.gradients(u2i1, xi1)[0]
        u2i1_y = tf.gradients(u2i1, yi1)[0]

        u2i1_xx = tf.gradients(u2i1_x, xi1)[0]
        u2i1_yy = tf.gradients(u2i1_y, yi1)[0]

        # Average value (Required for enforcing the average solution along the interface)
        uavgi1 = (u1i1 + u2i1)/2

        # Residuals
        f1 = u1_xx + u1_yy - (tf.exp(x1) + tf.exp(y1))
        f2 = u2_xx + u2_yy - (-1 + tf.exp(x2) + tf.exp(y2))

        # Residual continuity conditions on the interfaces
        fi1 = (u1i1_xx + u1i1_yy - (tf.exp(xi1) + tf.exp(yi1))) - (u2i1_xx + u2i1_yy - (-1 + tf.exp(xi1) + tf.exp(yi1)))

        return f1, f2, fi1, uavgi1, u1i1, u2i1

    def setup_training(self, X_ub, ub, X_f1, X_f2, X_fi, u_fi):
        """Setup training data and loss functions."""
        # Training data
        self.x1_ub = X_ub[:,0:1]
        self.y1_ub = X_ub[:,1:2]
        self.ub1 = ub

        self.x_f1 = X_f1[:,0:1]
        self.y_f1 = X_f1[:,1:2]
        self.x_f2 = X_f2[:,0:1]
        self.y_f2 = X_f2[:,1:2]

        self.x_fi1 = X_fi[:,0:1]
        self.y_fi1 = X_fi[:,1:2]
        self.u_fi = u_fi

        # TensorFlow placeholders
        self.x1_ub_tf = tf.placeholder(tf.float64, shape=[None, self.x1_ub.shape[1]])
        self.y1_ub_tf = tf.placeholder(tf.float64, shape=[None, self.y1_ub.shape[1]])

        self.x_f1_tf = tf.placeholder(tf.float64, shape=[None, self.x_f1.shape[1]])
        self.y_f1_tf = tf.placeholder(tf.float64, shape=[None, self.y_f1.shape[1]])
        self.x_f2_tf = tf.placeholder(tf.float64, shape=[None, self.x_f2.shape[1]])
        self.y_f2_tf = tf.placeholder(tf.float64, shape=[None, self.y_f2.shape[1]])

        self.x_fi1_tf = tf.placeholder(tf.float64, shape=[None, self.x_fi1.shape[1]])
        self.y_fi1_tf = tf.placeholder(tf.float64, shape=[None, self.y_fi1.shape[1]])

        # Predictions
        self.ub1_pred = self.net_u1(self.x1_ub_tf, self.y1_ub_tf)
        self.ub2_pred = self.net_u2(self.x_f2_tf, self.y_f2_tf)

        self.f1_pred, self.f2_pred, self.fi1_pred, \
            self.uavgi1_pred, self.u1i1_pred, self.u2i1_pred = \
            self.net_f(self.x_f1_tf, self.y_f1_tf, self.x_f2_tf, self.y_f2_tf, self.x_fi1_tf, self.y_fi1_tf)

        # Loss functions
        self.loss1 = self.multiplier*tf.reduce_mean(tf.square(self.ub1 - self.ub1_pred)) \
                        + self.multiplier*tf.reduce_mean(tf.square(self.u_fi - self.u1i1_pred)) \
                        + tf.reduce_mean(tf.square(self.f1_pred)) + 1*tf.reduce_mean(tf.square(self.fi1_pred))\
                        + self.multiplier*tf.reduce_mean(tf.square(self.u1i1_pred-self.uavgi1_pred))

        self.loss2 = tf.reduce_mean(tf.square(self.f2_pred)) + 1*tf.reduce_mean(tf.square(self.fi1_pred))\
                        + self.multiplier*tf.reduce_mean(tf.square(self.u_fi - self.u2i1_pred)) \
                        + self.multiplier*tf.reduce_mean(tf.square(self.u2i1_pred-self.uavgi1_pred))

        # Optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer(0.0008)
        self.train_op_Adam1 = self.optimizer_Adam.minimize(self.loss1)
        self.train_op_Adam2 = self.optimizer_Adam.minimize(self.loss2)

        # Initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train_step(self):
        """Perform one training step."""
        tf_dict = {self.x1_ub_tf: self.x1_ub, self.y1_ub_tf: self.y1_ub,
                   self.x_f1_tf: self.x_f1, self.y_f1_tf: self.y_f1,
                   self.x_f2_tf: self.x_f2, self.y_f2_tf: self.y_f2,
                   self.x_fi1_tf: self.x_fi1, self.y_fi1_tf: self.y_fi1}

        self.sess.run(self.train_op_Adam1, tf_dict)
        self.sess.run(self.train_op_Adam2, tf_dict)

    def predict(self, X_star1, X_star2):
        """Make predictions on test data."""
        u_star1 = self.sess.run(self.ub1_pred, {self.x1_ub_tf: X_star1[:,0:1], self.y1_ub_tf: X_star1[:,1:2]})
        u_star2 = self.sess.run(self.ub2_pred, {self.x_f2_tf: X_star2[:,0:1], self.y_f2_tf: X_star2[:,1:2]})
        return u_star1, u_star2

    def get_loss(self):
        """Get current loss values."""
        tf_dict = {self.x1_ub_tf: self.x1_ub, self.y1_ub_tf: self.y1_ub,
                   self.x_f1_tf: self.x_f1, self.y_f1_tf: self.y_f1,
                   self.x_f2_tf: self.x_f2, self.y_f2_tf: self.y_f2,
                   self.x_fi1_tf: self.x_fi1, self.y_fi1_tf: self.y_fi1}

        loss1_value = self.sess.run(self.loss1, tf_dict)
        loss2_value = self.sess.run(self.loss2, tf_dict)
        return loss1_value, loss2_value
