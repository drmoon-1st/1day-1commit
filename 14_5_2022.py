import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import rand
from sklearn.linear_model import LinearRegression
import random
data_x = []
data_y = []

for i in range(0, 100):
    data_x.append(i)
    data_y.append(data_x[i]*random.random() + random.random())
# ax.scatter(data_x, data_y)
# plt.show()

data_x = np.array(data_x)
data_y = np.array(data_y)
reg = LinearRegression().fit(data_x[:, np.newaxis], data_y)
pred =  reg.predict(data_y[:, np.newaxis])
print(reg.coef_[0], reg.intercept_)