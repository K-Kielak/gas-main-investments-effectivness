from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class MLModel(ABC):

    def __init__(self, input_size, output_size, name=None, dtype=tf.float32):
        self.name = name

        # Initialize norm params variables
        self._output_size = output_size
        with tf.name_scope(self.name):
            self._are_norm_params_set = tf.Variable(False, dtype=tf.bool)
            self._in_norm_params = tf.Variable([0], dtype=dtype,
                                               validate_shape=False)
            self._out_norm_params = tf.Variable([0], dtype=dtype,
                                                validate_shape=False)

        # Initialize placeholders
        self._inputs = tf.placeholder(shape=[None, input_size], dtype=dtype)
        self._labels = tf.placeholder(shape=[None, output_size], dtype=dtype)

    @abstractmethod
    def train(self, inputs, labels, session):
        raise NotImplementedError('Method train of the model not implemented')

    @abstractmethod
    def calculate_average_distance(self, inputs, labels, session):
        raise NotImplementedError('Method calculate_average_distance '
                                  'of the model not implemented')

    @abstractmethod
    def predict(self, inputs, session):
        raise NotImplementedError('Method predict of the '
                                  'model not implemented')

    def normalize_inputs(self, inputs, sess):
        if not sess.run(self._are_norm_params_set):
            return inputs

        inp_norm_params = sess.run(self._in_norm_params)

        norm_inputs = []
        for inp in inputs:
            norm_inp = [feat*scale_coeff - mean_change
                        for feat, (scale_coeff, mean_change)
                        in zip(inp, inp_norm_params)]
            norm_inputs.append(norm_inp)

        return np.array(norm_inputs)

    def denormalize_outputs(self, outputs, sess):
        if not sess.run(self._are_norm_params_set):
            return outputs

        out_norm_params = sess.run(self._out_norm_params)

        norm_outputs = []
        for out in outputs:
            norm_out = [(out_feat + mean_change) / scale_coeff
                        for out_feat, (scale_coeff, mean_change)
                        in zip(out, out_norm_params)]
            norm_outputs.append(norm_out)

        return np.array(norm_outputs)

    def set_norm_params(self, params, sess):
        """
        Sets normalization parameters (scale_coefficient, mean_change)
        so if model was trained on normalized data for convergence
        reasons, it remembers how to normalize inputs or denormalize
        outputs if it's necessary.
        """
        assign_ops = [
                      tf.assign(self._are_norm_params_set, True),
                      tf.assign(self._in_norm_params,
                                params[:-self._output_size],
                                validate_shape=False),
                      tf.assign(self._out_norm_params,
                                params[-self._output_size:],
                                validate_shape=False)
        ]
        [sess.run(op) for op in assign_ops]


def initialize_weights(shape, dtype=tf.float32):
    """
    Initializes trainable weights variable tensor to the [-1:1]
    range according to truncated normal distribution
    """
    weights_distro = tf.truncated_normal(shape, stddev=0.5, dtype=dtype)
    return tf.Variable(weights_distro, dtype=dtype)


def initialize_biases(shape, dtype=tf.float32):
    """
    Initializes trainable bias variable tensor to the [0:1]
    range according to truncated normal distribution
    """
    biases_distro = tf.truncated_normal(shape, mean=0.5,
                                        stddev=0.25, dtype=dtype)
    return tf.Variable(biases_distro, dtype=dtype)
