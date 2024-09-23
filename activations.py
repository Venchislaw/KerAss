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
}

def relu_b(z):
    return z > 0

diff_act_map = {
    "linear": lambda z: 1,
    "relu": relu_b,
    "sigmoid": sigmoid,
    "softmax": softmax,
}