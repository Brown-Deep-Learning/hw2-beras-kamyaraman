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
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        means = np.mean((y_pred - y_true) ** 2)
        return np.mean(means)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred_gradient = [2 * (self.inputs[0] - self.inputs[1])]
        y_true_gradient = [0] 
        return [y_pred_gradient, y_true_gradient]


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
