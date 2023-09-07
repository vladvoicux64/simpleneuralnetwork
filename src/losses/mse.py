import numpy as np


def mse(y_true, y_predicted):
    return np.mean(np.power(y_true - y_predicted, 2))


def mse_derivative(y_true, y_predicted):
    return 2 * (y_predicted - y_true) / y_true.size
