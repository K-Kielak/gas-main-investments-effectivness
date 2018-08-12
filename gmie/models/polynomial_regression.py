import numpy as np
import tensorflow as tf

from gmie.models.ml_model import MLModel, initialize_weights, initialize_biases
from gmie.preprocessing import expand_to_polynomial


class PolynomialRegression(MLModel):
    def __init__(self, input_size, output_size, degree=1,
                 name=None, dtype=tf.float64):
        input_size = len(expand_to_polynomial(range(input_size), degree))
        super().__init__(input_size, output_size, name=name, dtype=dtype)
        self._degree = degree

        # Initialize the model
        with tf.name_scope(name):
            self._weights = initialize_weights([input_size, output_size],
                                               dtype=dtype)
            self._biases = initialize_biases([output_size], dtype=dtype)

        self._output = tf.matmul(self._inputs, self._weights) + self._biases

        # Calculate absolute distance for logging purposes
        distances = tf.abs(self._output - self._labels)
        self._average_distance = tf.reduce_mean(distances)

        # Calculate square error for training purposes
        square_errors = tf.square(self._output - self._labels)
        self._loss = tf.reduce_sum(square_errors)
        self._train_step = tf.train.AdamOptimizer().minimize(self._loss)

    def train(self, inputs, labels, session):
        session.run(self._train_step, feed_dict={
            self._inputs: self._prepare_data(inputs),
            self._labels: labels
        })

    def solve(self, inputs, labels, session):
        inputs = self._prepare_data(inputs)
        inputs_num = len(inputs)
        bias_input = np.ones([inputs_num, 1])
        inputs = np.append(bias_input, inputs, axis=1)

        solution = np.linalg.inv(np.matmul(inputs.T, inputs))  # (X^t * X)^-1
        solution = np.matmul(solution, inputs.T)  # (X^t * X)^-1 * X^t
        solution = np.matmul(solution, labels)  # (X^t * X)^-1 * X^t * y
        solved_biases = solution[0]
        solved_weights = solution[1:]

        # assign solved parameters
        session.run(self._biases.assign(solved_biases))
        session.run(self._weights.assign(solved_weights))

    def calculate_average_distance(self, inputs, labels, session):
        return session.run(self._average_distance, feed_dict={
            self._inputs: self._prepare_data(inputs),
            self._labels: labels
        })

    def predict(self, inputs, session):
        return session.run(self._output, feed_dict={
            self._inputs: self._prepare_data(inputs)
        })

    def _prepare_data(self, data):
        return [expand_to_polynomial(row, self._degree) for row in data]
