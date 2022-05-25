import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import SCORERS, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pylab as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
nb1 = GaussianNB()
nb1.fit(X_train, y_train)
print(nb1.score(X_test, y_test))
nb1 = BernoulliNB()
nb1.fit(X_train, y_train)
y_pred = nb1.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(f1_score(y_test, y_pred, average=None))
p = nb1.predict_proba(X_test[y_test != 2])
fpr, tpr, th = roc_curve(y_test[y_test != 2], p[:, 1])

f, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot(fpr, fpr)
plt.show()

X, y = load_breast_cancer(return_X_y=True)
X = X[:, (0, 1, 3)]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
g1 = GaussianNB()
g1.fit(X_train, y_train)
y_pred = g1.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average=None))
print(f1_score(y_test, y_pred, average=None))

filter = y_test != 0
score = g1.predict_proba(X_test)[:, 1]
fpr, tpr, th = roc_curve(y_test, score)
print(np.shape(y_test))
print(np.shape(score))
f, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot(fpr, fpr)
plt.show()
