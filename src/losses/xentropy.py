import numpy as np


def xentropy(y_true, y_predicted):
    return -np.sum(y_true * np.log(y_predicted + np.ones_like(y_predicted) * 10 ** -100))


def xentropy_derivative(y_true, y_predicted):
    return -y_true / (y_predicted + np.ones_like(y_predicted) * 10 ** -100)
