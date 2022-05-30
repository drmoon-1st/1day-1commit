# import sys
# while (1):
#     n, m = map(int, sys.stdin.readline().split())
#     if n==0 and m==0:
#         break
#     a1 = [int(sys.stdin.readline()) for _ in range(n)]
#     a2 = [int(sys.stdin.readline()) for _ in range(m)]        
#     cnt = 0
#     for i in a2:
#         front, rear = 0, n - 1
#         while front <= rear:
#             mid = (front + rear) // 2
#             if a1[mid] == i:
#                 cnt += 1
#                 break
#             elif a1[mid] > i:
#                 rear = mid - 1
#             else:
#                 front = mid + 1
#     print(cnt)

# from collections import defaultdict
# input = sys.stdin.readline
# while True:
#     cd = defaultdict(bool)
#     n, m = map(int, input().split())
#     cnt = 0
#     if n==0 and m==0:
#         break
#     for _ in range(n):
#         cd[int(input())] = True
#     for _ in range(m):
#         if cd[int(input())]:
#             cnt += 1
#     print(cnt)

# a = int(sys.stdin.readline())
# front = 1
# rear = a//2
# while True:
#     mid = (front + rear) // 2
#     if mid ** 2 == a:
#         break
#     elif mid ** 2 > a:
#         rear = mid -1
#     else:
#         front = mid + 1
# print(mid)

# import sys
# def fact(n):
#     if n <= 1:
#         return n
#     else:
#         return n * fact(n-1)
# n = str(fact(int(sys.stdin.readline())))
# cnt = 0
# p = 0
# n = n[::-1]
# if len(n) == 1:
#     print(0)
# else:
#     while True:
#         if n[p] == '0':
#             cnt += 1
#         else:
#             break
#         p += 1
#     print(cnt)

# import sys
# def fibo(n):
#     cnt = [0, 1]
#     if n == 0:
#         return cnt[::-1]
#     elif n == 1:
#         return cnt
#     else:
#         for i in range(2, n+1):
#             cnt.append(cnt[i-1] + cnt[i-2])
#         return cnt[i-1], cnt[i]
# for i in range(int(sys.stdin.readline())):
#     print(*fibo(int(sys.stdin.readline())))

##########

from cv2 import kmeans
import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import maxabs_scale
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import column
from yaml import load


####

class Node:
    def __init__(self, data, next=None) -> None:
        self.data = data
        self.next = next

    def insert(self, pos, val):
        pass

    def insert_first(self, val):
        pass

    def insert_last(self, val):
        pass

    def clear(self):
        pass

    def get(self, pos):
        pass

    def len(self):
        pass

    def is_empty(self):
        pass

    def is_full(self):
        pass

    def display(self):
        pass

#######

# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# kn = KNeighborsClassifier().fit(X_train, y_train)

# bag = BaggingClassifier(KNeighborsRegressor(), max_samples=0.5, max_features=0.5).fit(X_train, y_train)

# from sklearn.ensemble import RandomForestClassifier

# fr = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)

# from sklearn.ensemble import AdaBoostClassifier

# ab = AdaBoostClassifier(n_estimators=10).fit(X_train,y_train)

# import seaborn as sns

# data = sns.load_dataset('penguins')
# data.dropna(inplace=True)
# X = data[['body_mass_g', 'flipper_length_mm']]
# data[['species', 'island', 'sex']].astype('category').cat.codes
# from sklearn.cluster import KMeans
# km = KMeans(n_clusters=2).fit(X)
# print(km.labels_)
# print(data)