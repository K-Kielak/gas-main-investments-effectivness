import numpy as np


USED_DTYPE = np.float64
DATA_PATH = './data.txt'
FEATURES = ('Diameter', 'Model', 'Year', '?')
OUTPUTS = ('Price',)

features = []
labels = []
with open(DATA_PATH, 'r') as data_file:
    for line in data_file.readlines():
        datapoint = line.strip().split(';')
        if len(datapoint) != len(FEATURES) + len(OUTPUTS):
            raise IOError('Data in the data file doesn\'t match with specified data properties.'
                          'It contains {} values, whereas properties specified {} values.'
                          .format(len(datapoint), len(FEATURES) + len(OUTPUTS)))

        features.append(np.array(datapoint[:len(FEATURES)], dtype=USED_DTYPE))
        labels.append(np.array(datapoint[len(FEATURES):], dtype=USED_DTYPE))

print(features)
print(labels)