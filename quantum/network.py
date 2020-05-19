"""This module provides the neural network part of the implementation."""
import numpy as np

from quantum.data_utils import generate_dataset, iterate_minibatches
from quantum.quantum_utils import run_circuit

DIMENSION = 4
LABELS = {4: '1', 9: '0'}


def init_weights(dimension):
    """Generates dimension-1 weights for the neural network.

    Args:
        dimension: Amount of weights do generate.

    Returns:
        weights_: List of np.arrays with shape(dimension-1, 16).
    """
    weights_ = []
    for _ in range(dimension - 1):
        weight = np.random.random(4**2)  #TODO check init of wheights
        weights_.append(weight)
    return weights_


def loss(prediction, label, lambda_=0.234, eta=5.59):
    #TODO add train accuracy
    runs = 1024
    p_label = prediction.pop(LABELS[label], None) / runs
    p_max_not = max(prediction.values()) / runs
    return max(p_max_not - p_label + lambda_, 0)**eta


def batch_loss(batch):
    loss_ = 0
    x_batch, y_batch = batch
    for image, label in zip(x_batch, y_batch):
        prediction = run_circuit(image.flatten(), WEIGHTS)
        loss_ += loss(prediction, label)
    return loss_ / len(batch[0])


def train_epochs(x_train, y_train, batchsize=222, epochs=30):

    for epoch in range(epochs):
        for batch in iterate_minibatches(x_train, y_train, batchsize):
            b_loss = batch_loss(batch)
            #TODO update step


if __name__ == "__main__":
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = generate_dataset(DIMENSION,
                                                            filter_values=True,
                                                            value_true=4,
                                                            value_false=9)
    WEIGHTS = init_weights(DIMENSION**2)
    train_epochs(X_TRAIN, Y_TRAIN)
