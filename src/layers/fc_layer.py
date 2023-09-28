import numpy as np

from src.layers.layer import Layer


class FCLayer(Layer):
    def __init__(self, input_size, output_size, init_negative_weights=False):
        super().__init__()
        self.weights = np.random.rand(input_size, output_size) - init_negative_weights * 0.5
        self.biases = np.random.rand(1, output_size) - init_negative_weights * 0.5
        self.caches = [np.zeros_like(self.weights), np.zeros_like(self.biases)]

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    def compute_gradients(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)

        self.caches[0] -= learning_rate * weights_gradient
        self.caches[1] -= learning_rate * output_gradient
        return input_gradient

    def reset_caches(self):
        self.caches = [np.zeros_like(self.weights), np.zeros_like(self.biases)]

    def update_gradients(self, caches_average):
        self.weights += caches_average[0]
        self.biases += caches_average[1]
