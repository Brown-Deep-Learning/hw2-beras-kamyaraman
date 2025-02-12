import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):

    def __init__(self):
        self.one_hot = None
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.
    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique = np.unique(data)
        eye = np.eye(len(unique))
        self.one_hot = {}
        for i in range(len(unique)):
            self.one_hot[unique[i]] = eye[i]

    def forward(self, data):
        return np.array([self.one_hot[d] for d in data])

    def inverse(self, data):
        inverse = {(v,k) for k,v in self.one_hot.items()}
        return np.array([inverse[d] for d in data])
    
