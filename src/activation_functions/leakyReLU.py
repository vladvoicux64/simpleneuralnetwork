import numpy as np


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
