import numpy as np


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

# seems to work fine
