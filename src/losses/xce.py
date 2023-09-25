import numpy as np


def xce(y_true, y_predicted):
    return -np.mean(y_true * np.log(y_predicted + np.ones_like(y_predicted) * 10 ** -10))


def xce_derivative_softmax(y_true, y_predicted):
    return y_predicted - y_true


