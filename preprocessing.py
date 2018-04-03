import numpy as np


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