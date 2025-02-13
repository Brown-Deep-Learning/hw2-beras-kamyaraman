import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        ## given the output probabilities and true labels. 
        ## HINT: Argmax + boolean mask via '=='
        prediction = np.argmax(probs, axis=1)
        truth = np.argmax(labels, axis=1)
        return np.mean(prediction == truth)
