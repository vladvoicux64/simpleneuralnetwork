import numpy as np

from src.layers.layer import Layer


class MaxpoolingLayer(Layer):
    def __init__(self, input_shape, pool_shape):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        pool_height, pool_width = pool_shape
        self.input_shape = input_shape
        self.pool_shape = pool_shape
        self.output_shape = (input_depth, input_height // pool_height, input_width // pool_width)

    def forward_propagation(self, input):
        self.input = input
        output_depth, output_height, output_width = self.output_shape
        pool_height, pool_width = self.pool_shape
        self.output = np.zeros(self.output_shape)
        for i in range(output_depth):
            for j in range(output_height):
                for k in range(output_height):
                    self.output[i, j, k] = np.max(
                        input[i, j * pool_height: (j + 1) * pool_height, k * pool_width: (k + 1) * pool_width])
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        output_depth, output_height, output_width = self.output_shape
        pool_height, pool_width = self.pool_shape
        for i in range(output_depth):
            for j in range(output_height):
                for k in range(output_height):
                    max_area = self.input[i, j * pool_height: (j + 1) * pool_height,
                               k * pool_width: (k + 1) * pool_width]
                    maxj, maxk = np.unravel_index(np.argmax(max_area, axis=None), max_area.shape)
                    input_gradient[i, maxj, maxk] = output_gradient[i, j, k]
        return input_gradient
