import numpy as np


def leaky_relu(x, alpha = 0.01):
    return max(alpha * x, x)


def leaky_relu_derivative(x, alpha = 0.01):
    return np.where(x>0, 1, alpha)