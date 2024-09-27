"""
Layers Framework block.
Contains core layers for different purposes.
Source code by Venchislaw 2024.
MIT License.
"""

import numpy as np

from activations import activations_map, diff_act_map

class Dense:
    def __init__(self, n_neurons, activation="linear"):
        self.n_neurons = n_neurons
        self.activation = activation.strip().lower()

        self.weights = None
        self.bias = np.zeros((n_neurons, 1))
        self.X = None


    def forward(self, X):
        if not self.weights:
            # data is transposed
            n_features, n_samples  = X.shape
            self.weights = np.random.randn(self.n_neurons, n_features)
        self.X = X

        self.z = np.dot(self.weights, self.X) + self.bias
        self.a = activations_map[self.activation](self.z)
        return self.a


    def backward(self, output_grad, last=False, previous_w=None, Y=None, learning_rate=0.01):
        """if not last:
            da = np.dot(output_grad, previous_w)
            dz = da * diff_act_map[self.activation](self.z)
        else:
            da = Y - self.a
            dz = da * diff_act_map[self.activation](self.z)"""

        dz = output_grad * diff_act_map[self.activation](self.z)
        dw = np.dot(dz, self.X.T)  # Calculate gradient of weights correctly
        db = np.sum(dz, axis=1, keepdims=True)  # Correct gradient calculation for biases

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dz


"""
Reflection:

I'm sort of satisfied, but at the same time: "is it optimal solution?" scratches my brain.
I mean, I'm not quite sure it's a good way to explicitly check whether layer is the last one or not.
Anyway, I think that's all for today.

Tomorrow/Anytime else:
1) differentiate other activations (done)
2) add basic loss functions (done)
3) Implement Sequential API (basis)
4) Train Net with it.
"""
