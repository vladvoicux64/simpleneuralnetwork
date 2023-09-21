import numpy as np


def softmax(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


def softmax_derivative(x):
    sm = softmax(x).reshape(-1, 1)
    sm_derivative = np.diagflat(sm) - np.dot(sm, sm.T)
    return sm_derivative
