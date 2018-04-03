import tensorflow as tf


class LinearRegression(object):
    def __init__(self, features_num, outputs_num, name=None, dtype=tf.float64):
        self.name = name
        with tf.name_scope(name):
            self._inputs = tf.placeholder(shape=[None, features_num], name='inputs', dtype=dtype)
            self._labels = tf.placeholder(shape=[None, outputs_num], name='labels', dtype=dtype)

            weights = tf.Variable(tf.truncated_normal([features_num, outputs_num], stddev=0.1, dtype=dtype),
                                  name='weights')
            biases = tf.Variable(tf.truncated_normal([outputs_num], stddev=0.1, mean=0.2, dtype=dtype),
                                 name='biases')
            self._outputs = tf.matmul(self._inputs, weights) + biases
            self._average_distance = tf.reduce_mean(tf.abs(self._outputs - self._labels))

            square_errors = tf.square(self._outputs - self._labels)
            loss = tf.reduce_sum(square_errors)
            self._train_step = tf.train.AdamOptimizer().minimize(loss)

    def train(self, inputs, labels, session):
        session.run(self._train_step, feed_dict={
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
