import numpy as np
from scipy.special import expit


def mean_square_error(expected, predictons):
    return np.mean((expected - predictons)**2)


def euclidian_dis(a, b):
    return np.sqrt(np.sum((a - b)**2))


def accuracy(expected, predictions):
    return np.sum(expected == predictions) / len(expected)


def sigmod(x):
    return 1 / (1 + np.exp(-x))
