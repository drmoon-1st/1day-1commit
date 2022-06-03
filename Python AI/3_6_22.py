import sys
n, k = map(int, sys.stdin.readline().split())
data = [x for x in range(1, n+1)]
ans = []
cnt = k-1
while len(data) != 0:
    ans.append(data[cnt])
    data.remove(data[cnt])
    cnt += (k-1)
    while cnt >= len(data) and len(data) >= 1:
        cnt -= len(data)
ans[0] = '<'+str(ans[0])
ans[-1] = str(ans[-1])+'>'
for i in ans:
    if i == ans[-1]:
        print(i)
    else:
        print(i, end=', ')

import sys
a = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))

def get_ans(arr, a):
    ans = [1] * a
    for i in range(a):
        for j in range(i):
            if arr[i] > arr[j]:
                ans[i] = max(ans[i], ans[j]+1)
    return max(ans)
print(get_ans(arr, a))

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

print(df_train.columns.values)
'''['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']'''

print(df_train.head(10))

print(df_train.info())
print(df_test.info())

print(df_train.describe(include=['O']))

print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(df_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(df_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

g = sns.FacetGrid(df_train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show()

grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
plt.show()

grid = sns.FacetGrid(df_train, col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

grid = sns.FacetGrid(df_train, col='Survived', row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()

combine = [df_train, df_test]
print('Before', df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)
combine = [df_train, df_test]
print('Before', df_train.shape, df_test.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

print(pd.crosstab(df_train['Title'], df_train['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(df_test.columns.values)
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

grid = sns.FacetGrid(df_train, col='Sex', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
plt.show()

guess_ages = np.zeros((2,3))

print(df_train[df_train['Pclass']==4])

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()
            print(guess_df.head())
            age_guess = guess_df.median()
            print(age_guess)
            print('#####')