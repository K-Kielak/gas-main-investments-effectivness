from abc import ABC, abstractmethod

import tensorflow as tf


class MLModel(ABC):

    def __init__(self, input_size, output_size, name=None, dtype=tf.float32):
        self.name = name

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
