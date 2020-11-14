import numpy as np
import sys
sys.path.insert(1, './../')
import utils


class LogisticRegression:
    def __init__(self, learning_rate=0.001, itreates=1000):
        self.learning_rate = learning_rate
        self.itreates = itreates
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.biases = 0

        ### gradient decent
        for _ in range(self.itreates):
            ### calculate linear equation
            linear_model = np.dot(X, self.weights) + self.biases

            ### calcualte sigmod equation
            y_predicted = utils.sigmod(linear_model)

            ### dw = (1/N) * E(i=0 to n)(2X(y_predicted-y))
            dw = (1 / n_samples) * np.dot(2 * X.T, (y_predicted - y))

            ### db = (1/N) * E(i=0 to n)(y_predicted-y)
            db = (1 / n_samples) * np.sum(2 * (y_predicted - y))

            ### update weights & biases
            self.weights -= dw * self.learning_rate
            self.biases -= db * self.learning_rate

    def predit(self, X):
        linear_model = np.dot(X, self.weights) + self.biases
        y_predicted = utils.sigmod(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]