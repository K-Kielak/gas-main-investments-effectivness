from collections import defaultdict

import numpy as np
import tensorflow as tf

from gmie.config import *
from gmie.preprocessing import *


def main():
    datasets = read_data()
    train_inputs, train_labels, test_inputs, test_labels = \
        prepare_data(datasets)

    # Start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Solve solvable models
        for i, model in enumerate(SOLVABLE_MODELS):
            train_loss = model.calculate_average_distance(train_inputs,
                                                          train_labels, sess)
            test_loss = model.calculate_average_distance(test_inputs,
                                                         test_labels, sess)
            print(f'Train distance with {model.name} '
                  f'before solving: {train_loss}')
            print(f'Test distance with {model.name} '
                  f'before solving: {test_loss}')
            model.solve(train_inputs, train_labels, sess)
            train_loss = model.calculate_average_distance(train_inputs,
                                                          train_labels, sess)
            test_loss = model.calculate_average_distance(test_inputs,
                                                         test_labels, sess)
            print(f'Train distance with {model.name} '
                  f'after solving: {train_loss}')
            print(
                f'Test distance with {model.name} '
                f'after solving: {test_loss}')

        # Start training on trainable models
        for i in range(0, TRAIN_STEPS + 1):
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


def prepare_data(datasets):
    data = center_output_around(datasets, CENTRAL_DATAPOINT)
    data = np.array(data, dtype=DTYPE)
    train_data, test_data = get_test_train_data(data, TEST_DATA_SIZE)
    train_data, test_data, norm_parameters = normalize_train_test_data(
        train_data,
        test_data)

    train_inputs = train_data[:, :len(FEATURES)]
    train_labels = train_data[:, len(FEATURES):]
    test_inputs = test_data[:, :len(FEATURES)]
    test_labels = test_data[:, len(FEATURES):]
    return train_inputs, train_labels, test_inputs, test_labels


def read_data():
    datasets = defaultdict(list)
    with open(DATA_PATH, 'r') as data_file:
        for line in data_file.readlines():
            date, *datapoint = line.strip().split(';')
            datapoint = [float(d) for d in datapoint]

            if len(datapoint) != len(FEATURES) + len(OUTPUTS):
                raise IOError(f"Data in the data file doesn't match "
                              f"with specified data config. It "
                              f"contains {len(datapoint)} values, "
                              f"whereas config specified "
                              f"{len(FEATURES) + len(OUTPUTS)} values.")

            datasets[date].append(datapoint)

    return datasets


if __name__ == '__main__':
    main()
