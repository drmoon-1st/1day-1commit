from enum import Flag
import sys
# from collections import deque

# n, m, v = map(int, sys.stdin.readline().split())
# edge = [[] for x in range(n+1)]

# for i in range(m):
#     v1, v2 = map(int, sys.stdin.readline().split())
#     edge[v1].append(v2)
#     edge[v2].append(v1)

# for i in edge:
#     i.sort()

# def dfs(graph, start_node):
#     visited = []
#     need_visit = []
#     need_visit.append(start_node)

#     while need_visit:
#         node = need_visit.pop()
#         if node not in visited:
#             visited.append(node)
#             need_visit.extend(graph[node])
#     return visited    

# print(dfs(edge, v))

# import sys
# from collections import deque
# d = deque()
# for i in range(int(sys.stdin.readline())):
#     d.append(i+1)
# while d.__len__() > 1:
#     d.popleft()
#     d.append(d.popleft())
# print(d[0])

#####

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

# print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(df_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# g = sns.FacetGrid(df_train, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
# grid.add_legend()
# plt.show()

# grid = sns.FacetGrid(df_train, col='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# plt.show()

# grid = sns.FacetGrid(df_train, col='Survived', row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()

# combine = [df_train, df_test]
# print('Before', df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]
# print('Before', df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

# print(pd.crosstab(df_train['Title'], df_train['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
# print(df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# print(df_test.columns.values)
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# grid = sns.FacetGrid(df_train, col='Sex', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
# grid.add_legend()
# plt.show()

guess_ages = np.zeros((2,3))

# print(df_train[df_train['Pclass']==4])

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()
            # print(guess_df.head())
            age_guess = guess_df.median()
            # print(age_guess)
            # print('#####')
            guess_ages[i, j] = int(age_guess//0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)

df_train['AgeBand'] = pd.cut(df_train['Age'], 5)
# print(df_train[['AgeBand', 'Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

df_train = df_train.drop(['AgeBand'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# print(df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# print(df_train[['IsAlone', 'Survived']].groupby('IsAlone', as_index=False).mean())

df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

freq_port = df_train.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# print(df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

df_train['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)

df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = df_train.drop(['FareBand'], axis=1)
combine = [df_train, df_test]