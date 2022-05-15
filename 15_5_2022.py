import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import rand
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

data = load_diabetes()
# print(pd.DataFrame(data.data).dtypes)
data = data.data[:, 6:8]

# sns.heatmap(pd.DataFrame(data).corr(), annot=True)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1])

def gd_numpy(X, y, epochs=0, lr=0):
    w = 0.0
    b = 0.0

    w_list, b_list, loss_list = [], [], []

    for i in range(epochs):
        loss = np.mean((y-(w*X+b))**2)

        dw = -2 * np.mean((y-(w*X+b)) * X)
        db = -2 * np.mean(y-(w*X+b))

        w = w - dw * lr
        b = w - db * lr
        
        w_list.append(w)
        b_list.append(b)
        loss_list.append(loss)
    
    print('weight : %.4f' % w)
    print('bias : %.4f' % b)

    return w, b, w_list, b_list, loss_list

# epochs = 100000 
# learning_rate = 0.01

# w, b, w_list, b_list, loss_list = gd_numpy(X_train, y_train, epochs, learning_rate)
# print(w, b)

# y_pred = w * X_train + b
# train_loss = np.mean((y_train-y_pred)**2)
# print(train_loss)

def lr_sklearn(X, y):
    X_2d = X.reshape(X.shape[0], -1)

    reg = LinearRegression().fit(X_2d, y)

    w = reg.coef_[0]
    b = reg.intercept_

    print('w : %.4f' % w)
    print('b : %.4f' % b)

    return reg

# model = lr_sklearn(X_train, y_train)

def gd_sklearn(X, y, epochs, lr, alpha):
    X_2d = X[:, np.newaxis]

    reg = SGDRegressor(penalty='l2', alpha=alpha, max_iter=epochs, tol=1e-3, learning_rate= 'invscaling', eta0=lr, random_state=42)

    reg.fit(X_2d, y)

    w = reg.coef_[0]
    b = reg.intercept_

    print('w : %.4f' % w)
    print('b : %.4f' % b)

    return reg
gd_sklearn(X_train, y_train, 1000, 0.01, 0.001)
# f, ax = plt.subplots()
# ax.scatter(X_train, y_train)
# ax.plot(X_train, (model.coef_[0]*X_train+model.intercept_))
# plt.show()