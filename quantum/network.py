"""This module provides the neural network part of the implementation.
    Todos:
        * TODO: Hyperparameter tuning
        * TODO: Check Metrics
        * TODO: check matrix generation
"""
import operator
import pickle
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from quantum.utils import data_utils, quantum_utils

LABELS = {4: '1', 9: '0'}


class QuantumNetwork:
    """
    Class representing a Quantum Network.
    """
    def __init__(self, dimension, runs=1024, unitary_dim=4, efficient=True):
        """Initializes the networ parameters. Hyperparameters are set to zero.
            Weights are initialized randomly.

        Args:
            dimension (int): The problem size. The network has dimension²-1 unitaries.
            runs (int): Parameter for the circuit evaluation.
            unitary_dim (int): Dimensionality of unitaries: 4x4 -> 2 Qubits
                                                            8x8 -> 4 Qubits
            efficient (bool): Switch to use the efficient layout.
        """
        self.accuracies = []
        self.correct = 0
        self.losses = []
        self.efficient = efficient
        self.runs = runs
        if efficient:
            unitary_dim = 16
            self.qubits = 4
            self.weights = np.random.random(size=(dimension**2 - 3,
                                                  unitary_dim**2))
        else:
            self.qubits = dimension**2
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

    def spsa_batch_loss(self, batch, pertubation, track=False):
        """Calculates the batch loss given the prertubation and weights.

        Args:
            batch: The image and label batch.
            perubation: Pertubation for this batch.
            track (bool): Switch to track loss statistics.
        Returns:
            loss (float): the batch loss.
        """
        loss = 0
        x_batch, y_batch = batch
        weights_ = self.weights + pertubation
        for image, label in zip(x_batch, y_batch):
            if self.efficient:
                prediction = quantum_utils.run_efficient_circuit(
                    image.flatten(), weights_, runs=self.runs)
            else:
                prediction = quantum_utils.run_circuit(image.flatten(),
                                                       weights_,
                                                       runs=self.runs)
            # test = np.random.uniform(0, 256)
            # prediction = {'0': 256 - test, '1': test}
            # loss += self.spsa_loss(prediction, label, track)
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
        with Pool(processes=2) as pool:
            for epoch in range(epochs):
                self.correct = 0
                print(f'Epoch {epoch+1} out of {epochs}')
                count = 0
                alpha_k = self.spsa_a / (epoch + 1 + self.spsa_A)**self.spsa_s
                beta_k = self.spsa_b / (epoch + 1)**self.spsa_t
                for batch in data_utils.iterate_minibatches(x_train,
                                                            y_train,
                                                            batchsize,
                                                            shuffle=True):
                    start = time.time()
                    count += 1
                    pertubation = np.random.uniform(-1, 1, self.weights.shape)
                    b_loss = pool.starmap(
                        self.spsa_batch_loss,
                        [(batch, alpha_k * pertubation, True),
                         (batch, -alpha_k * pertubation)])
                    spsa_g = (b_loss[0] - b_loss[1]) / (2 * alpha_k)
                    self.losses.append(b_loss[0])
                    spsa_v = self.spsa_gamma * spsa_v - spsa_g * beta_k * pertubation
                    self.weights += spsa_v
                    end = time.time()
                    print(
                        f'Completed Batch {count} out of {x_train.shape[0]//batchsize +1} in {end-start} seconds'
                    )
                self.accuracies.append(self.correct / x_train.shape[0])

    def predict(self, image):
        """Predicts the label of an image.

        Args:
            image: The input image.

        Returns:
            prediction_label: The label according to the LABELS dict.
        """
        image = image.flatten()
        prediction = quantum_utils.run_circuit(image,
                                               self.weights,
                                               runs=self.runs)
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

    def save_model(self, filename='model.pickle'):
        with open('data' + filename, 'wb+') as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load_model(filename='model.pickle'):
        with open('data' + filename, 'rb') as file_:
            model = pickle.load(file_)
        return model


if __name__ == '__main__':
    DIMENSION = 4
    (X_TRAIN,
     Y_TRAIN), (X_TEST,
                Y_TEST) = data_utils.generate_dataset(DIMENSION,
                                                      filter_values=True,
                                                      value_true=4,
                                                      value_false=9)
    NETWORK = QuantumNetwork(DIMENSION, runs=256)
    NETWORK.set_spsa_hyperparameters()
    NETWORK.train_epochs(X_TRAIN, Y_TRAIN, epochs=1)
    test_count = 0
    for sample, label in zip(X_TEST, Y_TEST):
        if (NETWORK.predict(sample) == label):
            test_count += 1
    print(f'Test Accuracy: {test_count/X_TRAIN.shape[0]}')
    NETWORK.print_stats()
