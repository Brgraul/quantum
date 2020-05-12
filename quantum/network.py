"""This module provides the neural network part of the implementation"""
import numpy as np

from quantum.data_utils import generate_dataset
from quantum.utils import run_circuit

DIMENSION = 4


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


if __name__ == "__main__":
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = generate_dataset(DIMENSION)
    WEIGHTS = init_weights(DIMENSION**2)
    for index, image in enumerate(X_TRAIN):
        image = image.flatten()
        if index == 0:
            prediction = run_circuit(image, WEIGHTS)
        #TODO update step
    print(f' Prediction: {prediction}')
