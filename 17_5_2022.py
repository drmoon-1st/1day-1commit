# from sklearn.datasets import fetch_california_housing
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import  train_test_split
# from sklearn.linear_model import LogisticRegression
# '''val: medlnc, houseage/ target: averom'''

##########

##########


from heapq import nsmallest
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.50)

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=0.5, bootstrap=True)


bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score

# print(accuracy_score(y_pred, y_test))

tree_clf = DecisionTreeClassifier()

tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

from sklearn.metrics import accuracy_score

# print(accuracy_score(y_pred, y_test))

def plot_dataset(X, y, show=True):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "s")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "^")
    
    if show:
        plt.show()

def plot_decision_boundary(clf, axes=None):
    if axes is None:
        x0 = np.linspace(-3, 4, 100)
        x1 = np.linspace(-3, 4, 100)
    else:
        x0 = np.linspace(axes[0][0], axes[0][1], 100)
        x1 = np.linspace(axes[1][0], axes[1][1], 100)
        
    
    x0, x1 = np.meshgrid(x0, x1)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    
    y_pred = clf.predict(X_new).reshape(x0.shape)
    
    plt.contourf(x0, x1, y_pred, alpha=0.25)
    plt.show()

plt.figure(figsize=(6, 4))

plot_dataset(X, y, False)
plot_decision_boundary(tree_clf)

plt.show()

plt.figure(figsize=(6, 4))

plot_dataset(X, y, False)
plot_decision_boundary(bag_clf)

plt.show()

from sklearn.ensemble import RandomForestClassifier

# Random Forest 모델을 정의하고 모델을 학습시켜보기
#####################
rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
#####################

plt.figure(figsize=(6, 4))

plot_dataset(X, y, False)
plot_decision_boundary(rnd_clf)

plt.show()

print(rnd_clf.feature_importances_)

from sklearn.datasets import load_iris

# iris 데이터셋 전체를 학습한 이후에, 각 피처에 대한 중요도를 뽑아주세요.
# load_iris()['feature_names'] 활용
####################
data = load_iris()
y_iris = data.target
X_iris = data.data
# X_train, X_test, y_train, y_test = train_test_split(data, target)
rnd_clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16)
rnd_clf.fit(X_iris, y_iris)
for fit, score in zip(data['feature_names'], rnd_clf.feature_importances_):
    print("The importance score of feature %s is about %.3f" % (fit, score))
####################

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, learning_rate=0.5)

# 이전과 똑같이 학습 및 평가해주세요.
####################
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_train)

####################

plt.figure(figsize=(6, 4))

plot_dataset(X, y, False)
plot_decision_boundary(ada_clf)

plt.show()