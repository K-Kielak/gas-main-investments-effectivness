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
    train_data = data_copy[test_end_index:]
    test_data = data_copy[:test_end_index]
    return train_data, test_data


def normalize_train_test_data(train, test):
    """
    Normalizes 2 matrices of data, train and test making sure that normalization parameters are decided based
    only on train data to avoid leaking any information about test data to the model.
    :param train: Train data (as numpy array), second axis (axis=1 in numpy) should correspond to separate features.
    :param test: Test data (as numpy array), second axis (axis=1 in numpy) should correspond to separate features.
    :return: Normalized train data, normalized test data, and 2D array of normalization parameters. For each feature
    there are different normalization parameters and each row of normalization parameters consists of 2 variables:
    scale_coefficient and mean_change where normalized_value = scale_coefficient*normalized_value - mean_change
    (normalization parameters are decided purely based on the train data).
    """
    normalized_train_data = []
    normalization_parameters = []
    for column in train.T:
        normalized_column, scale_coefficient, mean_change = normalize_data_vector(column)
        normalized_train_data.append(normalized_column)
        normalization_parameters.append((scale_coefficient, mean_change))

    normalized_test_data = []
    for i, column in enumerate(test.T):
        scale_coeff = normalization_parameters[i][0]
        mean = normalization_parameters[i][1]
        normalized_column, _, _ = normalize_data_vector(column, scale_coefficient=scale_coeff, mean_change=mean)
        normalized_test_data.append(normalized_column)

    return np.array(normalized_train_data).T, np.array(normalized_test_data).T, normalization_parameters


def normalize_data_vector(data, scale_coefficient=None, mean_change=None):
    """
    Normalizes data based on given (or calculated based on the given data) normalization parameters
    :param data: 1D collection to normalize.
    :param scale_coefficient: Used to multiply all values (used to scale the data). If set to None calculated so all
    of the values after scaling perfectly fit into the range of 100.
    :param mean_change: Added to all of the values (used to change the mean of the data). If set to None calculated
    so the minimum value in the dataset after scaling is equal to 0.
    :return: Normalized data based on scale_coefficient and mean_chage in the same format as input data and
    where normalized_data = scale_coefficient*data - mean_change
    """
    min_value = min(data)
    if scale_coefficient is None:
        max_value = max(data)
        data_range = max_value - min_value
        scale_coefficient = 100 / data_range

    if mean_change is None:
        mean_change = (min_value * scale_coefficient)

    return [scale_coefficient*value - mean_change for value in data], scale_coefficient, mean_change
