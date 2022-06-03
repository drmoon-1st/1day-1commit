# '''merge sort'''
# a = [3, 4, 5, 7, 2, 9, 8, 0, 10, 0]
# arr2 = [None] * 10
# def merge(arr, left, mid, right):
#     l = left
#     k = left
#     j = mid + 1
#     while l <= mid and j <= right:
#         if arr[l] < arr[j]:
#             arr2[k] = arr[l]
#             k += 1
#             l += 1
#         else:
#             arr2[k] = arr[j]
#             k += 1
#             j += 1
            
#     if l > mid:
#         while j <= right:
#             arr2[k] = arr[j]
#             k += 1
#             j += 1
#     else:
#         while l <= mid:
#             arr2[k] = arr[l]
#             k += 1
#             l += 1
#     for i in range(left, right+1):
#         arr[i] = arr2[i]


# def merge_sort(arr, left, right):
#     if left < right:
#         mid = (left + right) // 2
#         merge_sort(arr, left, mid)
#         merge_sort(arr, mid+1, right)
#         merge(arr, left, mid, right)

# merge_sort(a, 0, len(a)-1)
# print(a)

a = [3, 4, 5, 7, 2, 9, 8, 0, 10, 7]

def heap_sort(arr):
    n = len(arr)
    for i in range(n):
        c = i
        while c != 0:
            r = (c-1)//2
            if  arr[r] < arr[c]:
                arr[r], arr[c] = arr[c], arr[r]
            c = r
    # print(arr)
    
    for i in range(n-1, -1, -1):
        arr[0], arr[i] = arr[i], arr[0]
        r = 0
        c = 1
        while c<i:
            c = 2*r+1
            if c<i-1 and arr[c] < arr[c+1]:
                c += 1
            if c<i and arr[r] < arr[c]:
                arr[r], arr[c] = arr[c], arr[r]
            r = c
        # print(arr)
    # print(arr)

# heap_sort(a)
# print(a)

# import sys
# data = []
# for _ in range(int(sys.stdin.readline())):
#     a = int(sys.stdin.readline())
#     if a == 0:
#         if len(data)>0:
#             data.pop()
#     else:
#         data.append(a)
# print(sum(data))        

###############

import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

df_train = pd.read_csv('train.csv', header=0)
df_test = pd.read_csv('test.csv', header=0)

# print(df_train.columns.values)
'''['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']'''

# print(df_train.head(10))

# print(df_train.info())
# print(df_test.info())

# print(df_train.describe(include=['O']))