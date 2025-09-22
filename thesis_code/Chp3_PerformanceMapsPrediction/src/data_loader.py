"""
Unified Data Loader for Efficiency Map Predictor

This module provides a unified interface for loading both efficiency and power factor data.
It removes hard-coded paths and makes data source selection configurable.
"""

import numpy as np
import glob
import h5py
from sklearn.metrics import confusion_matrix
import matlab.engine
import os
from pathlib import Path


class DataLoader:
    """
    Unified data loader for efficiency and power factor prediction.

    Args:
        data_path (str): Path to the data directory containing text files and HDF5 files
        data_type (str): Either 'efficiency' or 'powerfactor'
        batch_size (int): Batch size for data loading
        dim_x (int): Input dimension (default: 14)
        dim_y (int): Output dimension (default: 1)
    """

    def __init__(self, data_path, data_type='efficiency', batch_size=2, dim_x=14, dim_y=1):

        self.data_path = Path(data_path)
        self.data_type = data_type
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y

        # Set up paths based on data type
        if data_type == 'efficiency':
            self.text_path = self.data_path / 'TrainText'
            self.hdf5_path = self.data_path / 'datasetEffMapText_xReal.h5'
        elif data_type == 'powerfactor':
            self.text_path = self.data_path / 'TrainTextPf'
            self.hdf5_path = self.data_path / 'datasetPfMapText_xReal.h5'
        else:
            raise ValueError("data_type must be either 'efficiency' or 'powerfactor'")

        # Validate paths exist
        if not self.text_path.exists():
            raise FileNotFoundError(f"Text path not found: {self.text_path}")
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Initialize data structures
        self.names = []
        self.inp_dim = 0
        self.seq_len = []
        self.seq_len_sorted = []
        self.seq_len_order = []

        # Efficiency threshold for evaluation
        self.eff_threshold = 85

        # Get sequence lengths
        self.get_sequence_len()

        # Initialize MATLAB engine for visualization
        try:
            self.eng = matlab.engine.start_matlab()
            self.eng.addpath(str(self.data_path))
        except Exception as e:
            print(f"Warning: Could not initialize MATLAB engine: {e}")
            self.eng = None

    def data_feed(self, index):
        """
        Generate a batch of data for training/inference.

        Args:
            index (int): Starting index for batch

        Returns:
            tuple: (X_inp, X, Y, src_len) where:
                - X_inp: Input features from HDF5
                - X: Sequential input data
                - Y: Target values (efficiency or power factor)
                - src_len: Source sequence lengths
        """
        src_len = []
        max_seq_len = self.seq_len_sorted[index + self.batch_size]
        b = np.zeros([self.batch_size, max_seq_len, self.inp_dim])
        X_inp = np.empty((self.batch_size, self.dim_x, self.dim_y))

        for i in range(self.batch_size):

            file_number = self.seq_len_order[index + self.batch_size - i]
            file_seq_len = self.seq_len_sorted[index + self.batch_size - i]

            # Read text file
            file_name = self.text_path / f'{file_number + 1}.txt'
            with open(file_name) as f:
                a = f.readlines()

            b1 = a[2:]  # Skip header lines
            b2 = []
            for x in b1:
                x2 = x.split()
                b2.append([float(y) for y in x2])

            b3 = np.asarray(b2)
            b3 = np.where(b3 < 0, 0, b3)  # Replace negative values with 0
            b[i, 0:file_seq_len, :] = b3
            src_len.append(file_seq_len)

            # Read HDF5 data
            with h5py.File(self.hdf5_path, 'r') as hdf5_file:
                gp_train_xReal = hdf5_file['/DS_TrINP']
                a = gp_train_xReal[file_number]
                a = a[1:, :]  # Skip first row
                X_inp[i, :, :] = a

        X = b[:, :, 0:-1]  # Everything except target variable
        Y = b[:, :, -1]    # Target variable (efficiency or power factor)

        return X_inp, X, Y, src_len

    def get_sequence_len(self):
        """Get sequence lengths from all text files."""
        txt_files = list(self.text_path.glob("*.txt"))
        txt_counter = len(txt_files)

        for number in range(txt_counter):
            file_number = number + 1
            self.names.append(file_number)
            file_name = self.text_path / f'{file_number}.txt'

            with open(file_name) as f:
                a = f.readline()
                seq_size = [int(x) for x in a.split()]
                self.seq_len.append(seq_size[0])
                self.inp_dim = seq_size[1]

        # Sort sequences by length for efficient batching
        self.seq_len_order = sorted(range(len(self.seq_len)),
                                  key=lambda k: self.seq_len[k])
        self.seq_len_sorted = sorted(self.seq_len)

    def qualitative_measure(self, Nm, T, Tar, Pred, index):
        """
        Generate qualitative visualization using MATLAB.

        Args:
            Nm: Speed values
            T: Torque values
            Tar: Target values
            Pred: Predicted values
            index: File index for identification
        """
        if self.eng is None:
            print("MATLAB engine not available for visualization")
            return

        file_number = self.seq_len_order[index + self.batch_size]
        index_mat = np.ones(np.size(Nm)) * file_number

        com_stack = np.dstack((Nm, T, Tar, Pred, index_mat))
        com_stack = com_stack.squeeze(0)
        stack_mat = matlab.double(com_stack.tolist())

        self.eng.drawEffMap(stack_mat, nargout=0)

    def quantitative_measure(self, Tar, Pred, index):
        """
        Calculate quantitative metrics (confusion matrix).

        Args:
            Tar: Target values
            Pred: Predicted values
            index: File index

        Returns:
            confusion_matrix: Confusion matrix for binary classification
        """
        Tar = np.where(Tar > self.eff_threshold, 1, 0)
        Pred = np.where(Pred > self.eff_threshold, 1, 0)
        cm = confusion_matrix(Tar, Pred)

        # Save confusion matrix to file
        file_number = self.seq_len_order[index + self.batch_size]
        output_file = self.data_path / f'confusion_matrix_{file_number}.txt'

        with open(output_file, 'w') as f:
            f.write('TN FP FN TP \n')
            tn, fp, fn, tp = cm.ravel()
            f.write(f'{tn} {fp} {fn} {tp}')

        return cm

    def get_data_info(self):
        """Get information about the loaded data."""
        return {
            'data_type': self.data_type,
            'num_files': len(self.names),
            'input_dimension': self.inp_dim,
            'sequence_lengths': {
                'min': min(self.seq_len),
                'max': max(self.seq_len),
                'mean': np.mean(self.seq_len)
            },
            'data_path': str(self.data_path)
        }
