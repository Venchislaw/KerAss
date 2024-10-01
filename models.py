"""
Models Framework block.
Contains core model APIs.
Source code by Venchislaw 2024.
MIT License.
"""

from layers import Dense
import numpy as np
from losses import loss_map


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.optimizer = None
        self.loss = None

    def compile(self, optimizer="sgd", loss="mse"):
        self.optimizer = optimizer
        self.loss = loss_map[loss]

    def fit(self, X, y, epochs=1_000, verbosity_step=100, learning_rate=0.001):
        # forward pass:


        for epoch in range(epochs):
            output = X.T
            for layer in self.layers:
                output = layer.forward(output)


            loss_value = self.loss.forward(y, output)
            output_grad = self.loss.backward()
            # print("LOSS OUTPUT GRAD: ", output_grad)
            # laya = len(self.layers)
            for layer in reversed(self.layers):
                output_grad = layer.backward(output_grad, learning_rate)
                # print(f"Layer: {laya}: {output_grad}")
                # laya -= 1

            if epoch % verbosity_step == 0:
                print(f"Epoch: {epoch} | Loss: {loss_value}")

        return output

    def predict(self, X):
        output = X.T
        for layer in self.layers:
            output = layer.forward(output)
        return output
