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
        means = np.mean((y_pred - y_true) ** 2, axis = -1)
        total = np.mean(means)
        return Tensor(total)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred_gradient = (2 * (self.inputs[0] - self.inputs[1]))/self.inputs[0].shape[0]
        y_true_gradient = np.zeros(self.inputs[1].shape) 
        return [Tensor(y_pred_gradient), Tensor(y_true_gradient)]


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        return NotImplementedError

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        return NotImplementedError
