"""This module provides the neural network part of the implementation. """
import operator
import pickle
import time
from multiprocessing import Pool
import quimb as qu

import matplotlib.pyplot as plt
import numpy as np

from quantum.utils import data_utils, quantum_utils

LABELS = {4: '1', 9: '0'}

v_dual = qu(array, qtype='bra', normalized=True)

class QuantumNetwork:
    """
    Class representing a Quantum Network.
    """
    def __init__(self, dimension, runs=1024, unitary_dim=4, v=1):
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
        self.runs = runs
        self.v = v  # Bond dimension
        self.qubits = dimension**2
        self.steps = np.log2(self.qubits)
        self.gates_per_step = [] # Number of hermitian gates per step
        for i in range(1, self.steps + 1):
            self.gates_per_step.append(self.qubits // (2 ** i))
        self.weights = np.random.normal(size=(int(sum(self.gates_per_step)), 2**(4*v)))

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
                                 spsa_gamma=0.882,
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

            prediction = quantum_utils.run_circuit(image.flatten(),
                                                   weights_,
                                                   runs=self.runs)
            # test = np.random.uniform(0, 256)
            # prediction = {'0': 256 - test, '1': test}
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
                b_loss_1 = self.spsa_batch_loss(batch, alpha_k * pertubation, True)
                b_loss_2 = self.spsa_batch_loss(batch, -alpha_k * pertubation)
                spsa_g = (b_loss_1 - b_loss_2) / (2 * alpha_k)
                self.losses.append(b_loss_1)
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
                                               self.gates_per_step,
                                               runs=self.runs)
        prediciton = max(prediction.items(), key=operator.itemgetter(1))[0]
        return list(LABELS.keys())[list(LABELS.values()).index(prediciton)]

    def print_stats(self):
        """
        Generates images for network Accuracy and Loss.
        """
        plt.subplot(1, 2, 1)
        plt.plot(self.accuracies)
        plt.title('Network Stats')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.subplot(1, 2, 2)
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
    NETWORK = QuantumNetwork(DIMENSION, runs=512)
    NETWORK.set_spsa_hyperparameters()
    NETWORK.train_epochs(X_TRAIN, Y_TRAIN, epochs=5)
    # test_count = 0
    # for sample, label in zip(X_TEST, Y_TEST):
    #     if (NETWORK.predict(sample) == label):
    #         test_count += 1
    # print(f'Test Accuracy: {test_count/X_TRAIN.shape[0]}')
    NETWORK.print_stats()
    NETWORK.predict(X_TEST[0])
