import os

import numpy as np
import tensorflow as tf

from gmie.models.feedforward_nn import FeedforwardNN
from gmie.models.polynomial_regression import PolynomialRegression


PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# DATA CONFIG
# Absolute path to the data file.
DATA_PATH = os.path.join(PROJECT_ROOT, 'data.txt')
# Towards what datapoint data for all dates should be centered around
CENTRAL_DATAPOINT = (700, 0, 0)
# What features program should expect to find in the data file.
FEATURES = ('Diameter', 'Model', '?')
INPUT_SIZE = [len(FEATURES)]
# What labels program should expect in case of training in the data file.
OUTPUTS = ('Price',)
OUTPUT_SIZE = [len(OUTPUTS)]

# TRAINING CONFIG
# What part of the data should be set aside for testing
TEST_DATA_SIZE = 0.15
# How many training steps should be performed.
# Each train step models use whole training data.
TRAIN_STEPS = 50000
# How often to log training data
LOGGING_FREQUENCY = 10000
DTYPE = np.float32


# MODELS CONFIG
# Trainable models to use
TRAINING_MODELS = (
    FeedforwardNN(INPUT_SIZE + [18] + OUTPUT_SIZE, activation=tf.nn.leaky_relu,
                  name='feedforwad_nn_18_lrelu', dtype=DTYPE),
)

# Solvable models to use
SOLVABLE_MODELS = (
    PolynomialRegression(len(FEATURES), len(OUTPUTS), degree=2,
                         name='quadratic_regression', dtype=DTYPE),
)
