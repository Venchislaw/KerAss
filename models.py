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
        self.loss = loss

    def fit(self, X, y):
        # forward pass:
        output = X.T
        for layer in self.layers:
            output = layer.forward(output)
        for layer in reversed(self.layers + [loss_map[self.loss]]):
            # pozhe
        return output

x = np.random.randn(10_000, 20)
y = np.random.choice(2, 10_000)

seq = Sequential([
    Dense(10, "relu"),
    Dense(30, "relu"),
    Dense(1, "sigmoid")
])


output = seq.fit(x, y)
print(output)