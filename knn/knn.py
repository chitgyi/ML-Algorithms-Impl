import numpy as np
import sys

sys.path.insert(1, './../')
import utils
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def accuracy(self, expected, predictions):
        return utils.accuracy(expected, predictions)

    def fit(self, X, y):
        self.X_Train = X
        self.Y_Train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [
            utils.euclidian_dis(x, x_train) for x_train in self.X_Train
        ]

        ### take k neghtbors from sorted distances that return indices
        k_indices = np.argsort(distances)[:self.k]
        # print('K-Indices', k_indices)

        ### take 3 labels from 3 neightbors
        k_nearest_labels = [self.Y_Train[i] for i in k_indices]
        # print('K-Nearest-Labels', k_nearest_labels)

        ### take one from most common
        ### most_common return two argument (most_index, number_of_most_column)
        ### most_common(n) is take `n` row
        most_label, num_of_most_labels = Counter(k_nearest_labels).most_common(
            1)[0]
        # print('Most-Common', most_common)

        ### return label
        return most_label
