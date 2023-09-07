import numpy as np


def bce(y_true, y_predicted):
    return -np.mean(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))


def bce_derivative(y_true, y_predicted):
    return ((1 - y_true) / (1 - y_predicted) - y_true / y_predicted) / np.size(y_true)
