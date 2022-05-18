from sklearn.datasets import load_iris, fetch_california_housing, load_wine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression, SGDClassifier
from xgboost import train

# iris = load_iris()
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y)

# r1 = LogisticRegression(max_iter=2000)
# r1.fit(X_train, y_train)
# print(r1.score(X_test, y_test))

# def mysig(z):
#     return 1 / (1+np.exp(-z))

# d = r1.decision_function(X_test)
# p = r1.predict_proba(X_test)

# print(np.dot(X_test[0], r1.coef_[0])+r1.intercept_[0])
# print(mysig(d[0]), p[0])

house = load_wine()
X = house.data[:, (0, 9, 10)]
y = house.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=True)

# f, ax = plt.subplots()
# ax.scatter(X[:, 2], y)
# plt.show()

wine_clf = SGDClassifier(max_iter=13000)
wine_clf.fit(X_train, y_train)
y_pred = wine_clf.predict(X_test)
print(wine_clf.score(X_test, y_test))