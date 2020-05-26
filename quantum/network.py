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
        weight = np.random.normal(size=4**2)
        weights_.append(weight)
    return weights_


def loss(prediction, label, lambda_=0.234, eta=5.59):
    #TODO add train accuracy
    runs = 1024
    p_label = prediction.pop(LABELS[label], None) / runs
    p_max_not = max(prediction.values()) / runs
    return max(p_max_not - p_label + lambda_, 0)**eta


def batch_loss(batch, weights):
    loss_ = 0
    x_batch, y_batch = batch
    for image, label in zip(x_batch, y_batch):
        #prediction = run_circuit(image.flatten(), weights)
        prediction = {'0': 500, '1': 524}
        loss_ += loss(prediction, label)
    return loss_ / len(batch[0])


def train_epochs(x_train,
                 y_train,
                 batchsize=222,
                 epochs=30, 
                 a=28.0,
                 b=33.0,
                 A=74.1,
                 s=4.13,
                 t=0.658,
                 gamma=0.0882):

    v = np.zeros(WEIGHTS.shape)
    for epoch in range(epochs):
        alpha_k = a / (epoch + 1 + A)**s
        beta_k = b / (epoch + 1)**t
        for batch in iterate_minibatches(x_train, y_train, batchsize, shuffle=True):
            pertubation = np.random.uniform(-1, 1, WEIGHTS.shape[0])
            weights_1 = WEIGHTS + alpha_k * pertubation
            weights_2 = WEIGHTS - alpha_k * pertubation
            b_loss1 = batch_loss(batch, weights_1)
            b_loss2 = batch_loss(batch, weights_2)
            g = (b_loss1 - b_loss2) / (2 * alpha_k)
            v = gamma * v - g * beta_k * pertubation
            WEIGHTS += v


if __name__ == "__main__":
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = generate_dataset(DIMENSION,
                                                            filter_values=True,
                                                            value_true=4,
                                                            value_false=9)
    WEIGHTS = init_weights(DIMENSION**2)
    train_epochs(X_TRAIN, Y_TRAIN)
