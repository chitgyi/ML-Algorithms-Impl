from linear_regression import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [11], [13],
              [15], [17], [19]])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression(learning_rate=0.0005)
regressor.fit(X_train, y_train)
regression_line = regressor.predict(X)
predictions = regressor.predict(X_test)
mse = regressor.mse(y_test, predictions)
print(mse)
plt.scatter(X, y, label='Train Data')
plt.scatter(X_test, y_test, label='Test Data')
plt.plot(X, regression_line, label='Hypothesis', color='g')
plt.xlabel('Samples')
plt.ylabel('Labels')
plt.legend()
plt.show()
