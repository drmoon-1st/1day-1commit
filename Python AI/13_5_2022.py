from sklearn import datasets
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
bmi = diabetes.data[:, 2]
# print(bmi[:10])
target = diabetes.target

# f, ax = plt.subplots()
# ax.scatter(np.arange(442), bmi)

# sns.distplot(bmi)

def normal(x, mu=0, sigma=1):
    return math.exp(-(x-mu)**2)/(2*(sigma**2)/(math.sqrt(2*math.pi)*sigma))

xs = np.linspace(-5,5,442)
ys = np.array([normal(x) for x in xs])

# f, ax = plt.subplots()
# ax.plot(xs, ys)
# plt.show()

# import statsmodels.api as sm
# sm.qqplot(bmi)

def myline(x, a=1.0, b=1.0):
    return (a*x)+b

def back_prop(x, y, a=1.0, b=1.0):
    y_hat = a*x + b
    er = y-y_hat
    a = a+x*er
    b = b+1*er
    return a, b

def sse(tgt, model):
    return ((tgt-model)**2).sum()

def mse(tgt, model):
    return ((tgt-model)**2).mean()

a = 1.0
b = 1.0
lsta = []
for i in range(100):
    for bmii, tari in zip(bmi, target):
        a, b = back_prop(bmii, tari, a, b)
        lsta.append(a)

# f, ax = plt.subplots()
# ax.scatter(bmi, target)
# xs = np.linspace(-0.10, 0.15, 10)
# ax.plot(xs, myline(xs, a, b))
# plt.show()
# ans = 0
# for i in range(len(bmi)):
#     ans += ((target[i]-(a*bmi[i]+b))**2)**0.5
# e = target - myline(bmi, a, b)
# print(ans)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# reg = LinearRegression().fit(bmi[:, np.newaxis], target)
# # print(reg.coef_, reg.intercept_)

# errarray = target - myline(bmi, reg.coef_[0], reg.intercept_)
# sns.distplot(errarray)
# f, ax = plt.subplots()
# ax.plot(lsta)
# plt.show()

# print(sse(target, myline(bmi, reg.coef_[0], reg.intercept_)))
# print(mse(target, myline(bmi, reg.coef_[0], reg.intercept_)))
# print(mean_squared_error(target, myline(bmi, reg.coef_[0], reg.intercept_)))
# print(r2_score(target, myline(bmi, reg.coef_[0], reg.intercept_)))

def r2(tgt, model):
    return 1-sse(tgt, model)/sse(tgt, tgt.mean())
# print(r2(target, myline(bmi, reg.coef_[0], reg.intercept_)))

# bmi_train, bmi_test, tar_train, tar_test = train_test_split(bmi[:, np.newaxis], target)
# reg = LinearRegression().fit(bmi_train, tar_train)
# pred = reg.predict(bmi_test)
# print(pred[:10])
# print(tar_test[:10])

############
linn = datasets.load_linnerud()
# print(linn)
# f, ax = plt.subplots()
# sns.heatmap(pd.DataFrame(linn.data).corr(), annot=True)
# plt.show()

c = linn.data[:, 0]
t = linn.target[:, 0]
# sns.distplot(c)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(c[:, np.newaxis], t)
reg = LinearRegression().fit(x_train, y_train)
# print(reg.coef_[0], reg.intercept_)
# pred = reg.predict(x_test)
# print(pred[:10])
# print(y_test[:10])
# f, ax = plt.subplots()
# ax.scatter(c, t)
# ax.plot([2, 16], [(x*reg.coef_[0])+reg.intercept_ for x in [2, 16]])
# plt.show()
# print(r2_score(t, [(x*reg.coef_[0])+reg.intercept_ for x in c]))

target = linn.target
x_train, x_test, y_train, y_test = train_test_split(linn.data, target)
reg = LinearRegression().fit(x_train, y_train)
# print(reg.coef_[0], reg.intercept_)
pred = reg.predict(x_test)
print(pred[:10])
print(y_test[:10])
