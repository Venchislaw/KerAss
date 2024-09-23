import numpy as np
from activations import activations_map

class Dense:
    def __init__(self, n_neurons, activation="linear"):
        self.n_neurons = n_neurons
        self.activation = activation.strip().lower()

        self.weights = None
        self.bias = np.zeros((n_neurons, 1))
        self.X = None


    def forward_prop(self, X):
        if not self.weights:
            n_samples, n_features = X.shape
            self.weights = np.random.randn(self.n_neurons, n_features)
        self.X = X.T

        z = np.dot(self.weights, self.X) + self.bias
        a = activations_map[self.activation](z)
        return a


dense_layer = Dense(10, "ReLU")
res = dense_layer.forward_prop(np.random.randn(1_000, 30))
print(res.min())  # 0.0
