"""This module provides utility for dataset management."""
import collections

import numpy as np
import tensorflow as tf


def generate_dataset(size=4,
                     filter_contradicting=False,
                     filter_values=False,
                     value_true=3,
                     value_false=6):
    """Generates a customized MNIST dataset.

    Args:
        size: The dimension of resized images size x size.
        filter_contradicting: Switch to filter contradicting samples.
        filter_values: Switch to filter values.
        value_true: The number to for True labels.
        value_false: The number to for False labels.

    Returns:
        (x_train_resized, y_train): The training data.
        (x_test_resized, y_test): The testing data.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[...,
                              np.newaxis] / 255.0, x_test[...,
                                                          np.newaxis] / 255.0
    if filter_values:
        x_train, y_train = filter_dataset(x_train, y_train, value_true,
                                          value_false)
        x_test, y_test = filter_dataset(x_test, y_test, value_true,
                                        value_false)
    if filter_contradicting:
        x_train, y_train = filter_contradicting(x_train, y_train)
        x_test, y_test = filter_contradicting(x_test, y_test)

    x_train_resized = tf.image.resize(x_train, (size, size)).numpy()
    x_test_resized = tf.image.resize(x_test, (size, size)).numpy()
    return (x_train_resized, y_train), (x_test_resized, y_test)


def filter_dataset(x_data, y_data, value_true, value_false):
    """Filters examples from MNIST data to binary decision.

    Args:
        x_data: An array of ground trouth data.
        y_data: The array of labels.
        value_true: The number for True labels.
        value_false: The number for False labels.

    Returns:
        x_data: The filtered ground trouth array.
        y_data: The filtered labels.
    """
    keep = (y_data == value_true) | (y_data == value_false)
    x_data, y_data = x_data[keep], y_data[keep]
    # y_data = y_data == value_true
    return x_data, y_data


def remove_contradicting(x_data, y_data):
    """Removes contradicting examples from labeled data.

    Args:
        x_data: An array of ground trouth data.
        y_data: The array of labels

    Returns:
        new_x: The new ground trouth array.
        new_y: The new labels.
    """
    mapping = collections.defaultdict(set)
    for datapoint, label in zip(x_data, y_data):
        mapping[tuple(datapoint.flatten())].add(label)

    new_x = []
    new_y = []
    for datapoint, label in zip(x_data, y_data):
        labels = mapping[tuple(datapoint.flatten())]
        if len(labels) == 1:
            new_x.append(datapoint)
            new_y.append(list(labels)[0])
        else:
            pass
    return np.array(new_x), np.array(new_y)


def iterate_minibatches(x_data, y_data, batchsize, shuffle=False):
    """Iterator for minibatches from stackoverflow.

    Args:
        x_data: An array of ground trouth data.
        y_data: The array of labels
        batchsize: The desired batchsize.
        shuffle: Switch for shuffeling batches.

    Yields:
        A minibatch of the desired size.
    """
    if shuffle:
        indices = np.arange(x_data.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, x_data.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, x_data.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield x_data[excerpt], y_data[excerpt]