# activation layer inherited from base layer class
from layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_derivative = activation_derivative

    # passes input through activation function
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # givem dE/dY, returns dE/dX
    def backward_propagation(self, output_error, learning_rate):
        # no learnable parameters
        return self.activation_derivative(self.input) * output_error
