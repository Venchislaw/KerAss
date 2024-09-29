"""
Losses Framework block.
Contains core loss functions for different tasks.
Source code by Venchislaw 2024.
MIT License.
"""

import numpy as np


class Loss:
    loss_value = None
    y_true = a = None
    m = None

    def forward(self, y_true, a):
        self.y_true = y_true
        self.a = a
        assert y_true.shape == a.shape
        n, self.m = y_true.shape

    def backward(self):
        pass


# seems to work fine
# model example of loss block (It sounds funny for some professionals)
class MeanSquaredError(Loss):
    def forward(self, y_true, a):
        super().forward(y_true, a)
        self.loss_value = 1 / self.m * np.sum((y_true - a) ** 2)
        return self.loss_value
    
    def backward(self):
        da = 2 / self.m * (self.y_true - self.a)
        return da


# also works after some tests
class MeanAbsoluteError(Loss):
    def forward(self, y_true, a):
        super().forward(y_true, a)
        self.loss_value = 1 / self.m * np.sum(np.abs(y_true - a))
        return self.loss_value

    def backward(self):
        da = - 1 / self.m * np.sign(self.y_true - self.a)
        return da


# HOPE it works fine
class CategoricalCrossentropy(Loss):
    def forward(self, y_true, a):
        super().forward(y_true, a)
        a = np.clip(a, 1e-12, 1. - 1e-12)
        self.loss_value = 1 / self.m * -np.sum(y_true * np.log(a), axis=1)
        return self.loss_value

    def backward(self):
        da = 1 / self.m * (self.a - self.y_true)
        # da = ((1 - self.y_true) / (1 - self.a) - self.y_true / self.a) / np.size(self.y_true)
        return da


# kinda dumb, as I don't have separate SparseCat and BinCat class
# and I have different keys mapped to the same loss
loss_map = {
    "mse": MeanSquaredError(),
    "mean_squared_error": MeanSquaredError(),
    "meansquarederror": MeanSquaredError(),
    "mae": MeanAbsoluteError(),
    "mean_absolute_error": MeanAbsoluteError(),
    "meanabsolutederror": MeanAbsoluteError(),
    "categorical": CategoricalCrossentropy(),
    "cat_crossentropy": CategoricalCrossentropy(),
    "categorical_crossentropy": CategoricalCrossentropy(),
    "sparse_categorical": CategoricalCrossentropy(),
    "sparse_cat_entropy": CategoricalCrossentropy(),
    "sparse_categorical_crossentropy": CategoricalCrossentropy(),
    "binary_crossentropy": CategoricalCrossentropy(),
    "bce": CategoricalCrossentropy()
}
