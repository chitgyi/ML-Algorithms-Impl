import numpy as np


def mean_square_error(expected, predictons):
    return np.mean((expected - predictons)**2)


def euclidian_dis(a, b):
    return np.sqrt(np.sum((a - b)**2))


def accuracy(expected, predictons):
    return np.sum(expected == predictons) / len(expected)