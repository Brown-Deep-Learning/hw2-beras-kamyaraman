import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []


################################################################################
## Intermediate Activations To Put Between Layers

class LeakyReLU(Activation):

    ## TODO: Implement for default intermediate activation.

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        relu = np.where(x > 0, x, self.alpha * x)
        return Tensor(relu)
    def get_input_gradients(self) -> list[Tensor]:
        """
        Leaky ReLu backpropagation!
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        gradient = np.where(self.inputs[0] > 0, 1, self.alpha) 
        return [Tensor(gradient)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J

class ReLU(LeakyReLU):
    ## GIVEN: Just shows that relu is a degenerate case of the LeakyReLU
    def __init__(self):
        super().__init__(alpha=0)


################################################################################
## Output Activations For Probability-Space Outputs

class Sigmoid(Activation):
    
    ## TODO: Implement for default output activation to bind output to 0-1

    def forward(self, x) -> Tensor:

        return Tensor(1 / (1 + np.exp(-x)))

    def get_input_gradients(self) -> list[Tensor]:
        """
        To see what methods/variables you have access to, refer to the cheat sheet.
        Hint: Make sure not to mutate any instance variables. Return a new list[tensor(s)]
        """
        return [Tensor(np.exp(-self.inputs[0]) / (1 + np.exp(-self.inputs[0]))**2)]

    def compose_input_gradients(self, J):
        return self.get_input_gradients()[0] * J


class Softmax(Activation):
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    ## TODO [2470]: Implement for default output activation to bind output to 0-1
    def __init__(self):
        self.gradients = None

    def forward(self, x):
        self.inputs = x
        """Softmax forward propagation!"""
        ## stable version
        stable = x - np.max(x)
        exps = np.exp(stable)
        outs = exps/np.sum(exps)
        return Tensor(outs)
 

    def get_input_gradients(self):
        """Softmax input gradients! Using np.outer and np.fill_diagonal is helpful."""
        x, y = self.inputs + self.outputs
        if(len(x.shape) == 1):
            x = x.reshape(1, -1)
        bn, n = x.shape
        grad = np.zeros(shape=(bn, n, n), dtype=x.dtype)
        np.outer(-y, x, out=grad)
        np.fill_diagonal(grad, y * (1 - y))
        return [Tensor(grad)]