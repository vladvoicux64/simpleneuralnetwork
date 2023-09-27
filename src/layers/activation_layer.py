import numpy as np

from src.activation_functions.softmax import softmax
from src.layers.layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        if self.activation == softmax:
            return np.dot(output_gradient, self.activation_derivative(self.input).T)
        else:
            return self.activation_derivative(self.input) * output_gradient
