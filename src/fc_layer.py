# fully connected layer inherited from base layer class
from layer import Layer
import numpy as np


class FCLayer(Layer):
    # number of input resp. output neurons
    def __init__(self, input_size, output_size, init_negative_weights = False):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - init_negative_weights * 0.5
        self.biases = np.random.rand(1, output_size) - init_negative_weights * 0.5

    # Y = WX + B
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    # given dE/dY, computes dE/dW and dE/dB and returns input_error = dE/dX
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBiases = output_error

        # param. update
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error
        return input_error
