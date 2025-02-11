import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        self.y_pred = y_pred
        self.y_true = y_true
        means = np.mean((y_pred - y_true) ** 2, axis=0)
        return np.mean(means)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred_gradient = [2 * (self.y_pred - self.y_true)]
        y_true_gradient = [0] 
        return y_pred_gradient, y_true_gradient


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
