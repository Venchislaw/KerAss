"""
Losses Framework block.
Contains core loss functions for different tasks.
Source code by Venchislaw 2024.
MIT License.
"""

import numpy as np

# seems to work fine
# model example of loss block (It sounds funny for some professionals)
class MeanSquaredError:
    loss_value = None
    y_true = a = None
    m = None

    def forward(self, y_true, a):
        self.y_true = y_true
        self.a = a
        assert y_true.shape == a.shape
        n, self.m = y_true.shape
        self.loss_value = 1 / self.m * np.sum((y_true - a) ** 2)
        return self.loss_value
    
    def backward(self):
        da = 2 / self.m * (self.y_true - self.a)
        return da


class MeanAbsoluteError:
    loss_value = None
    y_true = a = None
    m = None

    def forward(self, y_true, a):
        self.y_true = y_true
        self.a = a
        assert y_true.shape == a.shape
        n, self.m = y_true.shape
        self.loss_value = 1 / self.m * np.sum(np.abs(y_true - a))
        return self.loss_value

    def backward(self):
        da = - 1 / self.m * np.sign(self.y_true - self.a)
        return da
