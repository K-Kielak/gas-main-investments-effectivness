import numpy as np
import tensorflow as tf
from models.feedforward_nn import FeedforwardNN
from models.linear_regression import LinearRegression

DATA_PATH = './data.txt'
FEATURES = ('Diameter', 'Model', 'Year', '?')
OUTPUTS = ('Price',)

USED_DTYPE = np.float64
TEST_DATA_SIZE = 0.15  # What part of the whole data should be set aside for testing
TRAIN_STEPS = 100000  # How many train steps to perform, at the moment each train step uses whole training data
LOGGING_FREQUENCY = 10000  # How often to log training data


def get_test_train_data(data, test_size):
    """
    Splits the data into 2 separate datasets, train set and test set
    :param data: Data to split
    :param test_size: What part of the whole data should be taken for the test data
    :return: Data sampled randomly into test and train data.
    """
    data_copy = data[:]  # copy to avoid mutation of data parameter and make the whole method immutable
    np.random.shuffle(data_copy)
    test_end_index = int(test_size * len(data))
    test_data = data_copy[:test_end_index]
    train_data = data_copy[test_end_index:]
    return test_data, train_data


def normalize_data(data, min_bound=0, max_bound=100):
    """
    Normalizes data.
    :param data: 1D collection to normalize.
    :param min_bound: Minimal bound for the data normalization.
    :param max_bound: Maximal bound for the data normalization.
    :return: Normalized data in between min_bound and max_bound, in the same format as input data.
    """
    if min_bound >= max_bound:
        raise ValueError('Minimal bound ({}) can\'t be lower than maximal bound ({})'.format(min_bound, max_bound))

    bound_range = max_bound - min_bound
    min_value = min(data)
    max_value = max(data)
    data_range = max_value - min_value
    scale_coefficient = bound_range / data_range
    mean_change = min_bound - (min_value * scale_coefficient)
    return [scale_coefficient*value + mean_change for value in data]


data = []
with open(DATA_PATH, 'r') as data_file:
    for line in data_file.readlines():
        datapoint = line.strip().split(';')
        if len(datapoint) != len(FEATURES) + len(OUTPUTS):
            raise IOError('Data in the data file doesn\'t match with specified data properties.'
                          'It contains {} values, whereas properties specified {} values.'
                          .format(len(datapoint), len(FEATURES) + len(OUTPUTS)))

        data.append(datapoint)


data = np.array(data, dtype=USED_DTYPE)
data = np.array([normalize_data(column) for column in data.T])
inputs = data[:len(FEATURES)].T
labels = data[len(FEATURES):].T

test_inputs, train_inputs = get_test_train_data(inputs, TEST_DATA_SIZE)
test_labels, train_labels = get_test_train_data(labels, TEST_DATA_SIZE)

models = (
    LinearRegression(len(FEATURES), len(OUTPUTS), name='linear_regression', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [720], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='overfitting_feedforward', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [90], len(OUTPUTS), activation=tf.nn.relu,
                  is_regression=True, name='feedforwad_nn_90_relu', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [45], len(OUTPUTS), activation=tf.nn.relu,
                  is_regression=True, name='feedforwad_nn_45_relu', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [18], len(OUTPUTS), activation=tf.nn.relu,
                  is_regression=True, name='feedforwad_nn_18_relu', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [90], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='feedforwad_nn_90_lrelu', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [45], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='feedforwad_nn_45_lrelu', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [18], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='feedforwad_nn_18_lrelu', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [90], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='feedforwad_nn_90_sigmoid', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [45], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='feedforwad_nn_45_sigmoid', dtype=USED_DTYPE),
    FeedforwardNN(len(FEATURES), [18], len(OUTPUTS), activation=tf.nn.leaky_relu,
                  is_regression=True, name='feedforwad_nn_18_sigmoid', dtype=USED_DTYPE)
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0, TRAIN_STEPS):
        for model in models:
            if i % LOGGING_FREQUENCY == 0:
                train_loss = model.calculate_average_distance(train_inputs, train_labels, sess)
                test_loss = model.calculate_average_distance(test_inputs, test_labels, sess)
                print('Step {} out of {}'.format(i, TRAIN_STEPS))
                print('Training distance with {}: {}'.format(model.name, train_loss))
                print('Testing distance with {}: {}'.format(model.name, test_loss))

            model.train(train_inputs, train_labels, sess)
