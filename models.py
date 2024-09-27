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

    def fit(self, X, y):
        # forward pass:
        output = X.T
        for layer in self.layers:
            output = layer.forward(output)
            print(output.shape)

        loss_value = self.loss.forward(y, output)
        output_grad = self.loss.backward()
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)

        return output

x = np.random.randn(10_000, 20)
y = np.random.choice(2, 10_000)
y = y.reshape(1, -1)
print(y.shape)

seq = Sequential([
    Dense(10, "relu"),
    Dense(30, "relu"),
    Dense(1, "sigmoid")
])
seq.compile("sgd", "cat_crossentropy")
output = seq.fit(x, y)  # causes error (error in Dense/backprop) - fix it pls


"""
Problem of the day:
Fucking dimension mismatch.
When calculating dz on backward with elementwise multiplication of activation derivative with output grad.
Gotta go through backprop entirely on paper (math part) 
"""