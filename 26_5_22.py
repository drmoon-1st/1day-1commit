import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import SCORERS, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
X = X[:, (0, 1, 3)]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

dt = DecisionTreeClassifier(max_depth=7).fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)

print(y_test)
print(y_pred)

data = pd.read_csv("bean.csv", header=0)
X = data[["ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]]
y = data[["Class"]]
y = pd.Series(y.values.squeeze(), dtype='category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

dt = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=15)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(dt.score(X_test, y_test))
f, ax = plt.subplots(1,2, figsize=(17, 10))
ax[0].scatter(np.arange(0, len(y_pred)), y_test, color='red')
ax[1].scatter(np.arange(0, len(y_pred)), y_pred)
plt.show()
p = precision_score(y_test, y_pred, average=None)
print(p)
fpr, tpr, tr = roc_curve(y_test, dt.predict_proba(X_test)[:, 1], pos_label=1)
f, ax = plt.subplots()
ax.plot(fpr, tpr)
plt.show()