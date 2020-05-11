import collections

import numpy as np
import tensorflow as tf


def generate_dataset(size=4,
                     filter_contradicting=False,
                     filter_values=False,
                     value_true=3,
                     value_false=6):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[...,
                              np.newaxis] / 255.0, x_test[...,
                                                          np.newaxis] / 255.0
    if (filter_values):
        x_train, y_train = filter_dataset(x_train, y_train, value_true,
                                          value_false)
        x_test, y_test = filter_dataset(x_test, y_test, value_true,
                                        value_false)
    if (filter_contradicting):
        x_train, y_train = filter_contradicting(x_train, y_train)
        x_test, y_test = filter_contradicting(x_test, y_test)

    x_train_sized = tf.image.resize(x_train, (size, size)).numpy()
    x_test_small = tf.image.resize(x_test, (size, size)).numpy()
    return (x_train, y_train), (x_test, y_test)


def filter_dataset(x_data, y_data, value_true, value_false):
    keep = (y_data == value_true) | (y_data == value_false)
    x_data, y_data = x_data[keep], y_data[keep]
    y_data = y_data == value_true
    return x_data, y_data


def remove_contradicting(x_data, y_data):
    mapping = collections.defaultdict(set)
    for x, y in zip(x_data, y_data):
        mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for x, y in zip(x_data, y_data):
        labels = mapping[tuple(x.flatten())]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(list(labels)[0])
        else:
            pass
    return np.array(new_x), np.array(new_y)
