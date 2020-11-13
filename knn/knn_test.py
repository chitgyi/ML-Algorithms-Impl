from knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = KNN(k=5)
clf.fit(X_train, y_train)

### test model
predictions = clf.predict(X_test)
accuracy = clf.accuracy(y_test, predictions)
print('Accuracy', "{:2.2f}%".format(accuracy * 100))

## test only one data
result = clf.predict([X[0]])[0]
print('Output Label Index is ', result)