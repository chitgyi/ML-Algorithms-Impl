import numpy as np

np.seterr(divide='ignore', invalid='ignore')


### n_features == weights
class LinearRegression:
    def __init__(self, learning_rate=0.001, iterates=1000):
        self.learning_rate = learning_rate
        self.iterates = iterates
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        ### initalize values
        self.weights = np.zeros(n_features)
        self.biases = 0

        ### gradient decents
        for _ in range(self.iterates):
            ## y = wx + b
            y_predicted = np.dot(X, self.weights) + self.biases

            ### dw = 1/N * E(j=1 to n) 2x(y_predicted -y)
            dw = (1 / n_samples) * np.dot(2 * X.T, (y_predicted - y))

            ### db = 1/N * E(j=1 to n) 2(y_predicted -y)
            db = (1 / n_samples) * np.sum(2 * (y_predicted - y))

            ### w = w - learining_rate * dw
            self.weights -= self.learning_rate * dw

            ### b = b - leaninng_rate * db
            self.biases -= self.learning_rate * db
        return self

    def predict(self, X):
        return self.weights * X + self.biases
