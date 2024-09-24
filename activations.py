import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


activations_map = {
    "linear": lambda z: z,
    "relu": relu,
    "sigmoid": sigmoid,
    "softmax": softmax,
    "tanh": np.tanh,
}

def relu_b(z):
    return z > 0

def sigmoid_b(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax_b(z):
    s = softmax(z).reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def tanh_b(x):
    return 1.0 - np.tanh(x)**2

diff_act_map = {
    "linear": lambda z: 1,
    "relu": relu_b,
    "sigmoid": sigmoid_b,
    "softmax": softmax_b,
    "tanh": tanh_b
}