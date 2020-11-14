import numpy as np
import sys
sys.path.insert(1, './../')
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression

datasets = load_breast_cancer()
X, y = datasets.data, datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

regressor = LogisticRegression(learning_rate=0.0001)
regressor.fit(X, y)
predictions = regressor.predit(X_test)
print(regressor.mse(y_test, predictions))
print(predictions)
