import numpy as np


DATA_PATH = './data.txt'
FEATURES = ('Diameter', 'Model', 'Year', '?')
OUTPUTS = ('Price',)

USED_DTYPE = np.float64
TEST_DATA_SIZE = 0.15  # What part of the whole data should be set aside for testing


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


def normalize_data(data, min_bound=-1, max_bound=1):
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

print('test inputs')
print(test_inputs)
print('test labels')
print(test_labels)
print('train inputs')
print(train_inputs)
print('train_labels')
print(train_labels)
