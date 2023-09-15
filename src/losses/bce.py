import numpy as np


def bce(y_true, y_predicted):
    return -np.mean(y_true * np.log(y_predicted) + (np.ones_like(y_true) - y_true) * np.log(
        np.ones_like(y_predicted) - y_predicted))


def bce_derivative(y_true, y_predicted):
    return ((np.ones_like(y_true) - y_true) / (
                np.ones_like(y_predicted) - y_predicted) - y_true / y_predicted) / np.size(y_true)
