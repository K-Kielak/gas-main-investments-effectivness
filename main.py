import numpy as np


USED_DTYPE = np.float64
DATA_PATH = './data.txt'
FEATURES = ('Diameter', 'Model', 'Year', '?')
OUTPUTS = ('Price',)


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
data = np.array([normalize_data(column) for column in data.T]).T

print(data)
