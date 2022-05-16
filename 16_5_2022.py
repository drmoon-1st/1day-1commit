from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
import math
from sklearn.datasets import load_diabetes, load_linnerud
from pydoc import visiblename
from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import rand
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data = load_iris()
X = data.data[:, (2, 3)]
y = data.target

idx = (y == 0) + (y == 1)
X = X[idx]
y = y[idx]

svm_clf = SVC(kernel="linear").fit(X, y)
# print(svm_clf.coef_)
# print(svm_clf.intercept_)


def plot(w, b, xmin, xmax):
    x = np.linspace(xmin, xmax, 200)
    db = -w[0] / w[1] * x - b / w[1]

    margin = 1 / w[1]
    up = db + margin
    down = db - margin

    plt.plot(x, db, color='black')
    plt.plot(x, up, "--", color='black')
    plt.plot(x, down, "--", color='black')


plt.figure(figsize=(6, 4))

w = svm_clf.coef_[0]
b = svm_clf.intercept_[0]

plot(w, b, 0, 5.5)

plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "s")
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "^")
plt.ylim(-1, 3)
plt.show()

f, ax = plt.subplots()
ax.scatter(data[:, 0], target)
plt.show()

f, ax = plt.subplots()
ax.scatter(data[:, 1], target)
plt.show()

f, ax = plt.subplots()
ax.scatter(data[:, 2], target)
plt.show()

f, ax = plt.subplots()
ax.scatter(data[:, 3], target)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    data[:, 0], target, test_size=0.2, random_state=42)

epochs = 1000
learning_rate = 1e-3

w = 0.0
b = 0.0

X = data[:, [2, 3]]
y = target
idx = (y == 0) + (y == 1)
X = X[idx]
y = y[idx]

for i in range(epochs):
    y_pred = 1 / (1 + np.exp(-(w*X + b)))
    loss = -np.mean(y * np.log(y_pred) + (1-y) * np.log(1 - y_pred))

    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    w = w - dw * learning_rate
    b = b - db * learning_rate

print(w, b)

f, ax = plt.subplots()
y_pred = 1 / (1 + np.exp(-(w*X + b)))
ax.scatter(X, y_pred, color='red')
ax.axhline(y=0.5, color='orange', linestyle='--')
index1 = (y == 0) * (y_pred > 0.5)
index2 = (y == 1) * (y_pred > 0.5)
index3 = (y == 2) * (y_pred > 0.5)
index = index1 + index2 + index3
ax.scatter(X[index], y[index], color='orange')
plt.show()


def gd_sklearn(X, y, max_iter, C):
    X_2d = X[:, np.newaxis]

    reg = LogisticRegression(
        penalty='l2', max_iter=max_iter, C=C, tol=1e-3, solver='lbfgs', random_state=42)
    reg.fit(X_2d, y)

    w = reg.coef_
    b = reg.intercept_

    print(w)
    print(b)

    return reg


epochs = 1000
C = 0.1
model = gd_sklearn(X_train, y_train, epochs, C)
X2d = X_train[:, np.newaxis]
y_pred = model.predict(X2d)
print(y_pred)
print(len(X2d))

f, ax = plt.subplots()
ax.scatter(X_test, y_test)
ax.scatter(X2d, y_pred, color='red')
plt.show()

data = load_iris()
X = data.data[:, (2, 3)]
y = (data.target == 2)


svm_clf = Pipeline([('scalar', StandardScaler()), ('svc', LinearSVC(C=1))])
svm_clf.fit(X, y)

scalar = StandardScaler()
svm_clf1 = LinearSVC(C=1)
svm_clf2 = LinearSVC(C=100)

pip1 = Pipeline([('scalar', scalar), ('svc', svm_clf1)])
pip2 = Pipeline([('scalar', scalar), ('svc', svm_clf2)])

pip1.fit(X, y)
pip2.fit(X, y)

m, s = scalar.mean_, scalar.scale_

z0 = (X[:, 0] - m[0]) / s[0]
z1 = (X[:, 1] - m[1]) / s[1]

plt.figure(figsize=(6, 4))

plt.plot(z0[y == 1], z1[y == 1], "s")
plt.plot(z0[y == 0], z1[y == 0], "^")

w = svm_clf1.coef_[0]
b = svm_clf1.intercept_[0]

plot(w, b, 0, 5.5)
plt.ylim(-2, 2)
plt.show()


X, y = make_moons(n_samples=1000, noise=0.1)


def plot(X, y):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 's')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], '^')

    plt.show()


plot(X, y)

poly = Pipeline([('scale', StandardScaler()),
                ('svc', SVC(kernel='poly', degree=3, coef0=10, C=5))])
poly.fit(X, y)


diabetes = load_diabetes()
X = diabetes.data[:, 2:4]
target = diabetes.target


def normal(x, mu=0, sigma=1):
    return math.exp(-(x-mu)**2)/(2*(sigma**2)/(math.sqrt(2*math.pi)*sigma))


def myline(x, a=1.0, b=1.0):
    return (a*x)+b


def mypredict(X, a, b):
    return np.array([np.dot(X, a) + b])


def back_prop(x, y, a=1.0, b=1.0):
    y_hat = a*x+b
    err = y-y_hat
    a = a+x*(y-y_hat)
    b = b+1*(y-y_hat)
    return a, b


def sse(tgt, model):
    return ((tgt-model)**2).sum()


def mse(tgt, model):
    return ((tgt-model)**2).mean()


def r2(tgt, model):
    return 1-sse(tgt, model)/sse(tgt, tgt.mean())


X_train, X_test, y_train, y_test = train_test_split(X, target, shuffle=False)
reg = LinearRegression().fit(X_train, y_train)
reg_r = Ridge(alpha=0.5).fit(X_train, y_train)
reg_l = Lasso(alpha=0.5).fit(X_train, y_train)
print(r2_score(target, reg.predict(X)))
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))
print(reg.predict(X_test))
print(mypredict(X_test, reg.coef_, reg.intercept_))

linn = load_linnerud()
data = linn.data
target = linn.target[:, 0]
X_train = data
y_train = target

X_train, X_test, y_train, y_test = train_test_split(data, target)
linear = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=0.5).fit(X_train, y_train)
lasso = Lasso(alpha=0.5).fit(X_train, y_train)
print(ridge.score(X_test, y_test))

kf = KFold(n_splits=3)
c = 0
i = 0
for ti, vi in kf.split(X_train):
    xt = X_train[ti]
    xv = X_train[vi]
    yt = y_train[ti]
    yv = y_train[vi]
    reg = LinearRegression().fit(xt, yt)
    c += reg.coef_
    i += reg.intercept_

print(c/3, i/3)