import numpy as np
import tensorflow as tf
from models.feedforward_nn import FeedforwardNN
from models.linear_regression import LinearRegression
from preprocessing import get_test_train_data, normalize_data

DATA_PATH = './data.txt'
FEATURES = ('Diameter', 'Model', 'Year', '?')
OUTPUTS = ('Price',)

USED_DTYPE = np.float64
TEST_DATA_SIZE = 0.15  # What part of the whole data should be set aside for testing
TRAIN_STEPS = 100000  # How many train steps to perform, at the moment each train step uses whole training data
LOGGING_FREQUENCY = 10000  # How often to log training data

SOLVABLE_MODELS = (
    LinearRegression(len(FEATURES), len(OUTPUTS), name='linear_regression', dtype=USED_DTYPE),
)
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
data = np.array([normalize_data(column) for column in data.T]).T
train_data, test_data = get_test_train_data(data, TEST_DATA_SIZE)

train_inputs = train_data[:, :len(FEATURES)]
train_labels = train_data[:, len(FEATURES):]
test_inputs = test_data[:, :len(FEATURES)]
test_labels = test_data[:, len(FEATURES):]

# Start training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Solve solvable models
    for model in SOLVABLE_MODELS:
        train_loss = model.calculate_average_distance(train_inputs, train_labels, sess)
        test_loss = model.calculate_average_distance(test_inputs, test_labels, sess)
        print('Training distance with {} before solving: {}'.format(model.name, train_loss))
        print('Testing distance with {} before solving: {}'.format(model.name, test_loss))
        model.solve(train_inputs, train_labels, sess)
        train_loss = model.calculate_average_distance(train_inputs, train_labels, sess)
        test_loss = model.calculate_average_distance(test_inputs, test_labels, sess)
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