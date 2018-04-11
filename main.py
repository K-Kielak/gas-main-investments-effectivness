import itertools
import numpy as np
import tensorflow as tf
from models.feedforward_nn import FeedforwardNN
from models.linear_regression import LinearRegression
from preprocessing import get_test_train_data, normalize_train_test_data, expand_to_polynomial

DATA_PATH = './data.txt'
FEATURES = ('Diameter', 'Model', 'Year', '?')
OUTPUTS = ('Price',)

USED_DTYPE = np.float64
TEST_DATA_SIZE = 0.15  # What part of the whole data should be set aside for testing
TRAIN_STEPS = 500000  # How many train steps to perform, at the moment each train step uses whole training data
LOGGING_FREQUENCY = 10000  # How often to log training data


TRAINING_MODELS = (
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

quadratic_features = list(itertools.combinations_with_replacement(FEATURES, 2))
cubic_features = list(itertools.combinations_with_replacement(FEATURES, 3))

SOLVABLE_MODELS = (
    LinearRegression(len(FEATURES), len(OUTPUTS), name='linear_regression', dtype=USED_DTYPE),
    LinearRegression(len(FEATURES) + len(quadratic_features), len(OUTPUTS),
                     name='quadratic_regression', dtype=USED_DTYPE),
    LinearRegression(len(FEATURES) + len(quadratic_features) + len(cubic_features), len(OUTPUTS),
                     name='cubic_regression', dtype=USED_DTYPE)
)

# Each solvable model has to have associated it's own features (due to polynomial regressions etc.).
# Make sure read data is transformed appropriately and added to this array for each of the solvable
# models as a tuples (train_data, test_data)
data_for_solvables = []


# Read data
data = []
with open(DATA_PATH, 'r') as data_file:
    for line in data_file.readlines():
        datapoint = line.strip().split(';')
        if len(datapoint) != len(FEATURES) + len(OUTPUTS):
            raise IOError('Data in the data file doesn\'t match with specified data properties.'
                          'It contains {} values, whereas properties specified {} values.'
                          .format(len(datapoint), len(FEATURES) + len(OUTPUTS)))

        data.append(datapoint)

# Prepare data
data = np.array(data, dtype=USED_DTYPE)
train_data, test_data = get_test_train_data(data, TEST_DATA_SIZE)
train_data, test_data, normalization_parameters = normalize_train_test_data(train_data, test_data)

train_inputs = train_data[:, :len(FEATURES)]
train_labels = train_data[:, len(FEATURES):]
test_inputs = test_data[:, :len(FEATURES)]
test_labels = test_data[:, len(FEATURES):]

data_for_solvables.append((train_inputs, test_inputs))
data_for_solvables.append((expand_to_polynomial(train_inputs, 2), expand_to_polynomial(test_inputs, 2)))

# Start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Solve solvable models
    for i, model in enumerate(SOLVABLE_MODELS):
        train_loss = model.calculate_average_distance(data_for_solvables[i][0], train_labels, sess)
        test_loss = model.calculate_average_distance(data_for_solvables[i][1], test_labels, sess)
        print('Training distance with {} before solving: {}'.format(model.name, train_loss))
        print('Testing distance with {} before solving: {}'.format(model.name, test_loss))
        model.solve(data_for_solvables[i][0], train_labels, sess)
        train_loss = model.calculate_average_distance(data_for_solvables[i][0], train_labels, sess)
        test_loss = model.calculate_average_distance(data_for_solvables[i][1], test_labels, sess)
        print('Training distance with {} after solving: {}'.format(model.name, train_loss))
        print('Testing distance with {} after solving: {}'.format(model.name, test_loss))

    # Start training on trainable models
    for i in range(0, TRAIN_STEPS):
        for model in TRAINING_MODELS:
            if i % LOGGING_FREQUENCY == 0:
                train_loss = model.calculate_average_distance(train_inputs, train_labels, sess)
                test_loss = model.calculate_average_distance(test_inputs, test_labels, sess)
                print('Step {} out of {}'.format(i, TRAIN_STEPS))
                print('Training distance with {}: {}'.format(model.name, train_loss))
                print('Testing distance with {}: {}'.format(model.name, test_loss))

            model.train(train_inputs, train_labels, sess)