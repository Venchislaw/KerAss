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
            n_samples, n_features = X.shape
            self.weights = np.random.randn(self.n_neurons, n_features)
        self.X = X.T

        self.z = np.dot(self.weights, self.X) + self.bias
        self.a = activations_map[self.activation](self.z)
        return self.a


    def backward(self, output_grad, last=False, previous_w=None, Y=None, learning_rate=0.01):
        if not last:
            da = np.dot(output_grad, previous_w)
            dz = da * diff_act_map[self.activation](self.z)
        else:
            da = Y - self.a
            dz = da * diff_act_map[self.activation](self.z)

        dw = np.dot(dz, self.X.T)  # Calculate gradient of weights correctly
        db = np.sum(dz, axis=1, keepdims=True)  # Correct gradient calculation for biases

        # Update weights and biases
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return dz, self.weights

dense_layer = Dense(10, "ReLU")
res = dense_layer.forward(np.random.randn(1_000, 30))
print(res.min())  # 0.0
tmp = dense_layer.weights.copy()
grad, w = dense_layer.backward(np.random.randn(10, 1_000), last=True, Y=np.random.randn(10, 1_000))

print(tmp[0][0], dense_layer.weights[0][0])
