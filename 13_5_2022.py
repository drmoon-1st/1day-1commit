import re
from tarfile import TarError
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

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

a = 1.0
b = 1.0

for i in range(100):
    for bmii, tari in zip(bmi, target):
        a, b = back_prop(bmii, tari, a, b)

# f, ax = plt.subplots()
# ax.scatter(bmi, target)
# xs = np.linspace(-0.10, 0.15, 10)
# ax.plot(xs, myline(xs, a, b))
# plt.show()
ans = 0
for i in range(len(bmi)):
    ans += ((target[i]-(a*bmi[i]+b))**2)**0.5

print(ans)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# reg = LinearRegression().fit(bmi[:, np.newaxis], target)
# print(reg.coef_, reg.intercept_)

