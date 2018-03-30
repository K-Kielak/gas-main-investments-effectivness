import tensorflow as tf


class FeedforwardNN:
    def __init__(self, features_num, layers, outputs_num, activation=tf.nn.relu,
                 is_regression=False, name=None, dtype=tf.float64):
        self.name = name
        with tf.name_scope(name):
            self._inputs = tf.placeholder(shape=[None, features_num], name='inputs', dtype=dtype)
            self._labels = tf.placeholder(shape=[None, outputs_num], name='labels', dtype=dtype)

            weights, biases = _create_fnn_model([features_num] + layers + [outputs_num])
            self._outputs = _model_output(self._inputs, weights, biases, activation=activation)
            self._average_distance = tf.reduce_mean(tf.abs(self._outputs - self._labels))

            if is_regression:
                errors = tf.square(self._outputs - self._labels)
            else:
                errors = tf.nn.sigmoid_cross_entropy_with_logits(self._outputs, self._labels)
                self._outputs = tf.nn.sigmoid(self._outputs)

            self._loss = tf.reduce_mean(errors)
            self._train_step = tf.train.AdamOptimizer().minimize(self._loss)

    def train(self, inputs, labels, session):
        session.run([self._loss, self._train_step], feed_dict={
            self._inputs: inputs,
            self._labels: labels
        })

    def calculate_average_distance(self, inputs, labels, session):
        return session.run(self._average_distance, feed_dict={
            self._inputs: inputs,
            self._labels: labels
        })

    def predict(self, inputs, session):
        return session.run(self._outputs, feed_dict={
            self._inputs: inputs
        })


def _create_fnn_model(layer_sizes):
    weights = []
    biases = []
    for i in range(1, len(layer_sizes)):
        with tf.name_scope('layer_' + str(i)):
            weights.append(_initialize_weights(layer_sizes[i-1:i+1]))
            biases.append(_initialize_biases([layer_sizes[i]]))

    return weights, biases


def _initialize_weights(size):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1, dtype=tf.float64), name='weights')


def _initialize_biases(size):
    return tf.Variable(tf.truncated_normal(size, mean=0.2, stddev=0.1, dtype=tf.float64), name='biases')


def _model_output(inputs, weights, biases, activation=tf.nn.relu):
    if len(weights) != len(biases):
        raise ValueError('Number of bias layers ({}) and weight layers ({}) don\'t match'
                         .format(len(weights), len(biases)))

    layer_sum = tf.matmul(inputs, weights[0]) + biases[0]
    layer_output = activation(layer_sum)
    for i in range(1, len(weights)-1):
        layer_sum = tf.matmul(layer_output, weights[i]) + biases[i]
        layer_output = activation(layer_sum)

    return tf.matmul(layer_output, weights[-1]) + biases[-1]
