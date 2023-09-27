import autograd.numpy as np


def xce(y_true, y_predicted):
    return -np.sum(y_true * np.log(y_predicted + np.ones_like(y_predicted) * 10 ** -100))


def xce_derivative(y_true, y_predicted):
    return -y_true / (y_predicted + np.ones_like(y_predicted) * 10 ** -100)
