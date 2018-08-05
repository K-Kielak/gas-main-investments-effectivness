import itertools
import operator
from functools import reduce

import numpy as np


def get_test_train_data(data, test_size):
    """
    Randomly splits the data into 2 separate
    data sets, train set and test set
    """

    data_copy = data[:]  # copy the data to make the whole method immutable
    np.random.shuffle(data_copy)
    test_end_index = int(test_size * len(data))
    train_data = data_copy[test_end_index:]
    test_data = data_copy[:test_end_index]
    return train_data, test_data


def expand_to_polynomial(data, degree):
    """
    Expands given data to the polynomial degree of correlations
    (e.g. ([x1, x2], 2) to [x1, x2, x1^2, x1*x2, x2^2]).
    """
    expanded_data = [[] for _ in data]
    for d in range(1, degree+1):
        for i, row in enumerate(data):
            combinations = itertools.combinations_with_replacement(row, d)
            polynomials = [reduce(operator.mul, combination)
                           for combination in combinations]
            expanded_data[i].extend(polynomials)

    return np.array(expanded_data)


def normalize_train_test_data(train, test):
    """
    Normalizes 2 matrices of data, train and test
    :param train: Train data (as numpy array), second axis
    (axis=1 in numpy) should correspond to separate features.
    :param test: Test data (as numpy array), second axis
    (axis=1 in numpy) should correspond to separate features.
    :return: Normalized train data, normalized test data, and 2D
    array of normalization parameters. For each feature there are
    different normalization parameters and each row of normalization
    parameters consists of 2 variables:
    scale_coefficient and mean_change where
    normalized_value = scale_coefficient*normalized_value - mean_change
    (normalization parameters are decided purely based on the train
    data to avoid leaking any test information to the model)
    """
    norm_train_data = []
    norm_parameters = []
    for column in train.T:
        norm_column, scale_coeff, mean_change = normalize_data_vector(column)
        norm_train_data.append(norm_column)
        norm_parameters.append((scale_coeff, mean_change))

    norm_test_data = []
    for i, column in enumerate(test.T):
        scale_coeff = norm_parameters[i][0]
        mean = norm_parameters[i][1]
        norm_column, _, _ = normalize_data_vector(column,
                                                  scale_coeff=scale_coeff,
                                                  mean_change=mean)
        norm_test_data.append(norm_column)

    norm_train_data = np.array(norm_train_data).T
    norm_test_data = np.array(norm_test_data).T
    return norm_train_data, norm_test_data, norm_parameters


def normalize_data_vector(data, scale_coeff=None, mean_change=None):
    """
    Normalizes data vector based on given (or calculated
    based on the given data) normalization parameters
    :param data: 1D collection to normalize.
    :param scale_coeff: Used to scale all values (used to scale the
    data). If set to None calculated so all of the values after
    scaling perfectly fit into the range of 100.
    :param mean_change: Added to all of the values (used to change
    the mean of the data). If set to None calculated so the minimum
    value in the dataset after scaling is equal to 0.
    :return: Normalized data based on scale_coefficient and
    mean_chage in the same format as input data and
    where normalized_data = scale_coefficient*data - mean_change
    """
    min_value = min(data)
    if scale_coeff is None:
        max_value = max(data)
        data_range = max_value - min_value
        scale_coeff = 100 / data_range

    if mean_change is None:
        mean_change = (min_value * scale_coeff)

    scaled_vector = [scale_coeff*value - mean_change for value in data]
    return scaled_vector, scale_coeff, mean_change
