import tensorflow as tf

from gmie.models.ml_model import MLModel, initialize_weights, initialize_biases


class FeedforwardNN(MLModel):
    def __init__(self, network_shape, activation=tf.nn.relu,
                 name=None, dtype=tf.float32):
        super().__init__(network_shape[0], network_shape[-1],
                         name=name, dtype=dtype)

        # Model the network
        self._weights, self._biases = _set_up_parameters(network_shape,
                                                         dtype=dtype)
        self._output = _model_output(self._inputs, self._weights,
                                     self._biases, activation=activation)

        # Calculate absolute distance for logging purposes
        distances = tf.abs(self._output - self._labels)
        self._average_distance = tf.reduce_mean(distances)

        # Calculate square error for training purposes
        square_errors = tf.square(self._output - self._labels)
        self._loss = tf.reduce_sum(square_errors)
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
        return session.run(self._output, feed_dict={
            self._inputs: inputs
        })


def _set_up_parameters(network_shape, dtype=tf.float32):
    weights = []
    biases = []
    for i in range(1, len(network_shape)):
        weights_dims = network_shape[i - 1:i + 1]
        weights.append(initialize_weights(weights_dims, dtype=dtype))
        biases.append(initialize_biases([network_shape[i]], dtype=dtype))

    return weights, biases


def _model_output(inputs, weights, biases, activation=tf.nn.relu):
    if len(weights) != len(biases):
        raise AttributeError(f"Number of bias layers ({len(weights)}) "
                             f"and weight layers ({len(biases)}) don't match")

    layer_sum = tf.matmul(inputs, weights[0]) + biases[0]
    layer_output = activation(layer_sum)
    for i in range(1, len(weights)-1):
        layer_sum = tf.matmul(layer_output, weights[i]) + biases[i]
        layer_output = activation(layer_sum)

    return tf.matmul(layer_output, weights[-1]) + biases[-1]
