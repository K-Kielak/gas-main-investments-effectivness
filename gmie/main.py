import itertools
from collections import defaultdict

import numpy as np
import tensorflow as tf

from gmie.config import *
from gmie.models.feedforward_nn import FeedforwardNN
from gmie.models.linear_regression import LinearRegression
from gmie.preprocessing import *


TRAINING_MODELS = (
    FeedforwardNN(INPUT_SIZE + [720] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='overfitting_feedforward', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [90] + OUTPUT_SIZE, activation=tf.nn.relu,
                  name='feedforwad_nn_90_relu', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [45] + OUTPUT_SIZE, activation=tf.nn.relu,
                  name='feedforwad_nn_45_relu', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [18] + OUTPUT_SIZE, activation=tf.nn.relu,
                  name='feedforwad_nn_18_relu', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [90] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_90_lrelu', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [45] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_45_lrelu', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [18] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_18_lrelu', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [90] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_90_sigmoid', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [45] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_45_sigmoid', dtype=DTYPE),
    FeedforwardNN(INPUT_SIZE + [18] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_18_sigmoid', dtype=DTYPE)
)

quadratic_features = list(itertools.combinations_with_replacement(FEATURES, 2))
cubic_features = list(itertools.combinations_with_replacement(FEATURES, 3))

SOLVABLE_MODELS = (
    LinearRegression(len(FEATURES), len(OUTPUTS),
                     name='linear_regression', dtype=DTYPE),
    LinearRegression(len(FEATURES) + len(quadratic_features), len(OUTPUTS),
                     name='quadratic_regression', dtype=DTYPE),
    LinearRegression(len(FEATURES) + len(quadratic_features) + len(cubic_features),
                     len(OUTPUTS), name='cubic_regression', dtype=DTYPE)
)

# Read data
datasets = defaultdict(list)
with open(DATA_PATH, 'r') as data_file:
    for line in data_file.readlines():
        date, *datapoint = line.strip().split(';')
        datapoint = [float(d) for d in datapoint]

        if len(datapoint) != len(FEATURES) + len(OUTPUTS):
            raise IOError(f"Data in the data file doesn't match with "
                          f"specified data properties. It contains "
                          f"{len(datapoint)} values, whereas "
                          f"properties specified "
                          f"{len(FEATURES) + len(OUTPUTS)} values.")

        datasets[date].append(datapoint)


# Prepare data
data = center_output_around(datasets, CENTRAL_DATAPOINT)
data = np.array(data, dtype=DTYPE)
train_data, test_data = get_test_train_data(data, TEST_DATA_SIZE)
train_data, test_data, norm_parameters = normalize_train_test_data(train_data,
                                                                   test_data)

train_inputs = train_data[:, :len(FEATURES)]
train_labels = train_data[:, len(FEATURES):]
test_inputs = test_data[:, :len(FEATURES)]
test_labels = test_data[:, len(FEATURES):]

# Each solvable model has to have associated it's own features
# (due to polynomial regressions etc.). Make sure read data is
# transformed appropriately and added to this array for each
# of the solvable models as a tuples (train_data, test_data)
data_for_solvables = [
    (train_inputs, test_inputs),
    (expand_to_polynomial(train_inputs, 2),
     expand_to_polynomial(test_inputs, 2)),
    (expand_to_polynomial(train_inputs, 3),
     expand_to_polynomial(test_inputs, 3))
]

# Start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Solve solvable models
    for i, model in enumerate(SOLVABLE_MODELS):
        train_loss = model.calculate_average_distance(data_for_solvables[i][0],
                                                      train_labels, sess)
        test_loss = model.calculate_average_distance(data_for_solvables[i][1],
                                                     test_labels, sess)
        print(f'Train distance with {model.name} before solving: {train_loss}')
        print(f'Test distance with {model.name} before solving: {test_loss}')
        model.solve(data_for_solvables[i][0], train_labels, sess)
        train_loss = model.calculate_average_distance(data_for_solvables[i][0],
                                                      train_labels, sess)
        test_loss = model.calculate_average_distance(data_for_solvables[i][1],
                                                     test_labels, sess)
        print(f'Train distance with {model.name} after solving: {train_loss}')
        print(f'Test distance with {model.name} after solving: {test_loss}')

    # Start training on trainable models
    for i in range(0, TRAIN_STEPS):
        for model in TRAINING_MODELS:
            if i % LOGGING_FREQUENCY == 0:
                train_loss = model.calculate_average_distance(train_inputs,
                                                              train_labels,
                                                              sess)
                test_loss = model.calculate_average_distance(test_inputs,
                                                             test_labels,
                                                             sess)
                print(f'Step {i} out of {TRAIN_STEPS}')
                print(f'Train distance with {model.name}: {train_loss}')
                print(f'Test distance with {model.name}: {test_loss}')

            model.train(train_inputs, train_labels, sess)
