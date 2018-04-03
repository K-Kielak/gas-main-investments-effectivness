import numpy as np
import tensorflow as tf


class LinearRegression(object):
    def __init__(self, features_num, outputs_num, name=None, dtype=tf.float64):
        self.name = name
        with tf.name_scope(name):
            self._inputs = tf.placeholder(shape=[None, features_num], name='inputs', dtype=dtype)
            self._labels = tf.placeholder(shape=[None, outputs_num], name='labels', dtype=dtype)

            self._weights = tf.Variable(tf.truncated_normal([features_num, outputs_num], stddev=0.1, dtype=dtype),
                                        name='weights')
            self._biases = tf.Variable(tf.truncated_normal([outputs_num], stddev=0.1, mean=0.2, dtype=dtype),
                                       name='biases')
            self._outputs = tf.matmul(self._inputs, self._weights) + self._biases
            self._average_distance = tf.reduce_mean(tf.abs(self._outputs - self._labels))

            square_errors = tf.square(self._outputs - self._labels)
            loss = tf.reduce_sum(square_errors)
            self._train_step = tf.train.AdamOptimizer().minimize(loss)

    def train(self, inputs, labels, session):
        session.run(self._train_step, feed_dict={
            self._inputs: inputs,
            self._labels: labels
        })

    def solve(self, inputs, labels, sess):
        inputs_num = len(inputs)
        bias_input = np.ones([inputs_num, 1])
        inputs = np.append(bias_input, inputs, axis=1)
        solved_parameters = np.linalg.inv(np.matmul(inputs.T, inputs))  # (X^t * X)^-1
        solved_parameters = np.matmul(solved_parameters, inputs.T)  # (X^t * X)^-1 * X^t
        solved_parameters = np.matmul(solved_parameters, labels)  # (X^t * X)^-1 * X^t * y
        solved_biases = solved_parameters[0]
        solved_weights = solved_parameters[1:]

        # assign solved parameters
        sess.run(self._biases.assign(solved_biases))
        sess.run(self._weights.assign(solved_weights))

    def calculate_average_distance(self, inputs, labels, session):
        return session.run(self._average_distance, feed_dict={
            self._inputs: inputs,
            self._labels: labels
        })

    def predict(self, inputs, session):
        return session.run(self._outputs, feed_dict={
            self._inputs: inputs
        })
