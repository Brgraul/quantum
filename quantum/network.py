"""This module provides the neural network part of the implementation."""
import operator

import matplotlib.pyplot as plt
import numpy as np

from quantum.data_utils import generate_dataset, iterate_minibatches
from quantum.quantum_utils import run_circuit

LABELS = {4: '1', 9: '0'}


class QuantumNetwork:
    def __init__(self, dimension, runs=1024, unitary_dim=4):
        self.accuracies = []
        self.correct = 0
        self.losses = []
        self.qubits = dimension**2
        self.runs = runs
        self.weights = np.random.normal(size=(self.qubits, unitary_dim**2))

    def set_spsa_hyperparameters(self,
                                 a=28.0,
                                 b=33.0,
                                 A=74.1,
                                 s=4.13,
                                 t=0.658,
                                 gamma=0.0882,
                                 eta=5.59,
                                 lambda_=0.234):
        self.a = a
        self.b = b
        self.A = A
        self.s = s
        self.t = t
        self.gamma = gamma
        self.eta = eta
        self.lambda_ = lambda_

    def spsa_loss(self, prediction, label, track=False):
        p_max = max(prediction.values()) / self.runs
        p_label = prediction.pop(LABELS[label], None) / self.runs
        if (p_max == p_label and track):
            self.correct += 1
        p_max_not = max(prediction.values()) / self.runs
        return max(p_max_not - p_label + self.lambda_, 0)**self.eta

    def spsa_batch_loss(self, batch, weights_, track=False):
        loss = 0
        x_batch, y_batch = batch
        for image, label in zip(x_batch, y_batch):
            prediction = run_circuit(image.flatten(), weights_)
            #test = np.random.uniform(0,1024)
            #prediction = {'0': 1024-test, '1': test}
            loss += self.spsa_loss(prediction, label, track)
        return loss / len(batch[0])

    def train_epochs(self, x_train, y_train, batchsize=222, epochs=30):
        v = np.zeros(self.weights.shape)
        for epoch in range(epochs):
            self.correct = 0
            alpha_k = self.a / (epoch + 1 + self.A)**self.s
            beta_k = self.b / (epoch + 1)**self.t
            for batch in iterate_minibatches(x_train,
                                             y_train,
                                             batchsize,
                                             shuffle=True):
                pertubation = np.random.uniform(-1, 1, self.weights.shape)
                weights_1 = self.weights + alpha_k * pertubation
                weights_2 = self.weights - alpha_k * pertubation
                b_loss1 = self.spsa_batch_loss(batch, weights_1, True)
                b_loss2 = self.spsa_batch_loss(batch, weights_2)
                g = (b_loss1 - b_loss2) / (2 * alpha_k)
                self.losses.append(b_loss1)
                v = self.gamma * v - g * beta_k * pertubation
                self.weights += v
            self.accuracies.append(self.correct / x_train.shape[0])

    def predict(self, image):
        image = image.flatten()
        prediction = run_circuit(image, self.weights)
        prediciton = max(prediction.items(), key=operator.itemgetter(1))[0]
        return list(LABELS.keys())[list(LABELS.values()).index(prediciton)]

    def print_stats(self):
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
    network = QuantumNetwork(DIMENSION)
    network.set_spsa_hyperparameters()
    network.train_epochs(X_TRAIN, Y_TRAIN, epochs=5)
    example = network.predict(X_TEST[0])
    print(f'Prediction: {example}, Actual: {Y_TEST[0]}')
    network.print_stats()