"""This module provides the neural network part of the implementation.
    Todos:
        * TODO: Hyperparameter tuning
        * TODO: Save weights.
        * TODO: Check Metrics
        * TODO: Test Network.
"""
import operator

import matplotlib.pyplot as plt
import numpy as np

from quantum.data_utils import generate_dataset, iterate_minibatches
from quantum.quantum_utils import run_circuit

LABELS = {4: '1', 9: '0'}


class QuantumNetwork:
    """
    Class representing a Quantum Network.
    """


    def __init__(self, dimension, runs=1024, unitary_dim=4):
        """Initializes the networ parameters. Hyperparameters are set to zero.
            Weights are initialized randomly.

        Args:
            dimension (int): The problem size. The network has dimension²-1 unitaries.
            runs (int): Parameter for the circuit evaluation.
            unitary_dim (int): Dimensionality of unitaries: 4x4 -> 2 Qubits
                                                            8x8 -> 4 Qubits
        """
        self.accuracies = []
        self.correct = 0
        self.losses = []
        self.qubits = dimension**2
        self.runs = runs
        self.weights = np.random.normal(size=(self.qubits, unitary_dim**2))

        #spsa
        self.spsa_a = 0
        self.spsa_b = 0
        self.spsa_A = 0
        self.spsa_s = 0
        self.spsa_t = 0
        self.spsa_gamma = 0
        self.spsa_eta = 0
        self.spsa_lambda_ = 0

    def set_spsa_hyperparameters(self,
                                 spsa_a=28.0,
                                 spsa_b=33.0,
                                 spsa_A=74.1,
                                 spsa_s=4.13,
                                 spsa_t=0.658,
                                 spsa_gamma=0.0882,
                                 spsa_eta=5.59,
                                 spsa_lambda_=0.234):
        """Sets params for the spsa algorithm, defaults from the paper.

        Args:
            spsa_a (float): a.
            spsa_b (float): b.
            spsa_A (float): A.
            spsa_s (float): s.
            spsa_t (float): t.
            spsa_gamma (float): gamma.
            spsa_eta (float): eta.
            spsa_lambda_ (float): lambda.
        """
        self.spsa_a = spsa_a
        self.spsa_b = spsa_b
        self.spsa_A = spsa_A
        self.spsa_s = spsa_s
        self.spsa_t = spsa_t
        self.spsa_gamma = spsa_gamma
        self.spsa_eta = spsa_eta
        self.spsa_lambda_ = spsa_lambda_

    def spsa_loss(self, prediction, label, track=False):
        """Calculates the individual loss given the prediction and the label.

        Args:
            prediction: The prediction counts.
            label: Actual class.
            track (bool): Switch to track loss statistics.
        Returns:
            loss (float): the single loss.
        """
        p_max = max(prediction.values()) / self.runs
        p_label = prediction.pop(LABELS[label], None) / self.runs
        if (p_max == p_label and track):
            self.correct += 1
        p_max_not = max(prediction.values()) / self.runs
        return max(p_max_not - p_label + self.spsa_lambda_, 0)**self.spsa_eta

    def spsa_batch_loss(self, batch, weights_, track=False):
        """Calculates the batch loss given the prertubated weights.

        Args:
            batch: The image and label batch.
            weights_: Pertubated weights for this batch.
            track (bool): Switch to track loss statistics.
        Returns:
            loss (float): the batch loss.
        """
        loss = 0
        x_batch, y_batch = batch
        for image, label in zip(x_batch, y_batch):
            prediction = run_circuit(image.flatten(), weights_)
            #test = np.random.uniform(0,1024)
            #prediction = {'0': 1024-test, '1': test}
            loss += self.spsa_loss(prediction, label, track)
        return loss / len(batch[0])

    def train_epochs(self, x_train, y_train, batchsize=222, epochs=30):
        """Trains the network on the imageset according to the SPSA algorithm
            described in the paper.

        Args:
            x_train: The training image set.
            y_train: The training labels.
            batchsize (int): Number of images per batch.
            epochs (int): Number of epochs.
        """
        spsa_v = np.zeros(self.weights.shape)
        for epoch in range(epochs):
            self.correct = 0
            alpha_k = self.spsa_a / (epoch + 1 + self.spsa_A)**self.spsa_s
            beta_k = self.spsa_b / (epoch + 1)**self.spsa_t
            for batch in iterate_minibatches(x_train,
                                             y_train,
                                             batchsize,
                                             shuffle=True):
                pertubation = np.random.uniform(-1, 1, self.weights.shape)
                weights_1 = self.weights + alpha_k * pertubation
                weights_2 = self.weights - alpha_k * pertubation
                b_loss1 = self.spsa_batch_loss(batch, weights_1, True)
                b_loss2 = self.spsa_batch_loss(batch, weights_2)
                spsa_g = (b_loss1 - b_loss2) / (2 * alpha_k)
                self.losses.append(b_loss1)
                spsa_v = self.spsa_gamma * spsa_v - spsa_g * beta_k * pertubation
                self.weights += spsa_v
            self.accuracies.append(self.correct / x_train.shape[0])

    def predict(self, image):
        """Predicts the label of an image.

        Args:
            image: The input image.

        Returns:
            prediction_label: The label according to the LABELS dict.
        """
        image = image.flatten()
        prediction = run_circuit(image, self.weights)
        prediciton = max(prediction.items(), key=operator.itemgetter(1))[0]
        return list(LABELS.keys())[list(LABELS.values()).index(prediciton)]

    def print_stats(self):
        """
        Generates images for network Accuracy and Loss.
        """
        plt.subplot(2, 1, 1)
        plt.plot(self.accuracies)
        plt.title('Network Stats')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(self.losses)
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.show()


if __name__ == "__main__":
    DIMENSION = 4
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = generate_dataset(DIMENSION,
                                                            filter_values=True,
                                                            value_true=4,
                                                            value_false=9)
    NETWORK = QuantumNetwork(DIMENSION)
    NETWORK.set_spsa_hyperparameters()
    NETWORK.train_epochs(X_TRAIN, Y_TRAIN, epochs=5)
    EXAMPLE = NETWORK.predict(X_TEST[0])
    print(f'Prediction: {EXAMPLE}, Actual: {Y_TEST[0]}')
    NETWORK.print_stats()
