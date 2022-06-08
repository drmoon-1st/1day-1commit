import sys
def add(arr, val):
    if val not in arr:
        arr.append(val)

def remove(arr, val):
    if val in arr:
        arr.remove(val)

def check(arr, val):
    if val in arr:
        print(1)
        return 1
    print(0)
    return 0

def toggle(arr, val):
    if val in arr:
        arr.remove(val)
    else:
        arr.append(val)

def all(arr):
    arr = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}

def empty(arr):
    arr.clear()

arr = []
for i in range(int(sys.stdin.readline())):
    a = input()
    if 'add' in a:
        add(arr, int(a.split()[1]))
    elif 'remove' in a:
        remove(arr, int(a.split()[1]))
    elif 'check' in a:
        check(arr, int(a.split()[1]))
    elif 'toggle' in a:
        toggle(arr, int(a.split()[1]))
    elif 'all' in a:
        all(arr)
    else:
        empty(arr)

import sys
arr = []
for i in range(int(sys.stdin.readline())):
    p = sys.stdin.readline().rstrip()
    try:
        a, b = p.split()
        b = int(b)
    except:
        a = p    
    if a == 'add':
        if b not in arr:
            arr.append(b)
    elif a == 'remove':
        if b in arr:
            arr.remove(b)
    elif a == 'check':
        if b in arr:
            print(1)
        else:
            print(0)
    elif a == 'toggle':
        if b in arr:
            arr.remove(b)
        else:
            arr.append(b)
    elif a == 'all':
        arr = [x for x in range(1, 21)]
    else:
       arr.clear()

import sys
a, b = map(int, sys.stdin.readline().split())
arr = {}
for i in range(a):
    k = sys.stdin.readline().rstrip()
    arr[k] = i+1
    arr[str(i+1)] = k
for i in range(b):
    p = sys.stdin.readline().rstrip()
    if p.isdigit():
        print(arr[str(p)])
    else:
        print(arr[p])

import sys
a, b = map(int, sys.stdin.readline().split())
cnt = 0
d = {}
ans = []
for i in range(a):
    d[sys.stdin.readline().rstrip()] = True
for i in range(b):
    p = sys.stdin.readline().rstrip()
    if p in d:
        ans.append(p)
        cnt += 1
ans.sort()
print(cnt)
for i in ans:
    print(i)

import sys
a, b = map(int, sys.stdin.readline().split())
coin = []
for i in range(a):
    coin.append(sys.stdin.readline())
cnt = 0
for i in reversed(coin):
    if b >= int(i):
        p = b // int(i)
        b -= int(i)*p
        cnt += 1*p
    if b == 0:
        break
print(cnt)

import sys
n = int(sys.stdin.readline())
m = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
mincnt = abs(100-n)
for val in range(1000001):
    p = str(val)
    for i in range(len(p)):
        if int(p[i]) in arr:
            break
        elif i == len(p)-1:
            mincnt = min(mincnt, len(p)+abs(n-val))
print(mincnt)

import sys
n = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
arr.sort()
cnt = 0
for i in range(n):
    cnt += sum(arr[0:i+1])
print(cnt)

import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

train = pd.read_csv('labeledTrainData.tsv', header=0, sep='\t')
# print(train.head())

import sys
a = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
cnt = 0
for i in range(len(arr)-1):
    swap = 0
    for j in range(len(arr)-1-i):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
            cnt += 1
            swap = 1
    if swap == 0:
        break
print(cnt)

def merge_sort(arr):
    global ans
    cnt = 0
    if len(arr) < 2:
        return arr

    mid = len(arr) // 2
    low_arr = merge_sort(arr[:mid])
    high_arr = merge_sort(arr[mid:])

    merged_arr = []
    l = h = 0
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] <= high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
            ans += (mid+h-len(merged_arr))
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    return merged_arr

import sys
a = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
ans = 0
merge_sort(arr)
print(ans)

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dig = load_digits()
pca = PCA(n_components=10).fit_transform(dig.data)

X = pca
y = dig.target

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
log = LogisticRegression().fit(X_train, y_train)
pred = log.predict(X_test)
print(log.score(X_test, y_test))

def quick_sort(array, start, end):
    global ans
    if start >= end: return
    pivot = start
    left, right = start + 1, end
    while left <= right:
        swap = False
        while left <= end and array[left] <= array[pivot]:
            left += 1
        while right > start and array[right] >= array[pivot]:
            right -= 1
        if left > right: 
            array[right], array[pivot] = array[pivot], array[right]
            # swap = True
        else:
            array[right], array[left] = array[left], array[right]
            swap = True
        if swap == True:
            ans += 1
    quick_sort(array, start, right - 1)
    quick_sort(array, right + 1, end)

import sys
a = int(sys.stdin.readline())
arr = [int(sys.stdin.readline()) for x in range(a)]
ans = 1
quick_sort(arr, 0, a-1)
print(ans)

import sys
a = int(sys.stdin.readline())
arr = [(int(sys.stdin.readline()),x) for x in range(a)]
sarr = sorted(arr)
ans = []
for i in range(a):
    ans.append(sarr[i][1] - arr[i][1])
print(max(ans)+1)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data = pd.read_csv('winequality-white.csv', header=0, sep=';')
# print(data)

pca = PCA(n_components=5).fit_transform(data)

X = pca
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
log = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
print(log.score(X_test, y_test))

import sys
dic = {}
a, b = map(int, sys.stdin.readline().split())
for i in range(a):
    n1, n2 = sys.stdin.readline().split(' ')
    dic[n1] = n2
for i in range(b):
    print(dic[sys.stdin.readline().rstrip()], end='')

import sys
N = int(sys.stdin.readline())
arr = [0] * (N+1)
for i in range(1, N+1):
  arr[i] = i
for i in range(2, N+1):
  for j in range(1,  int((i**(1/2)))+1):
    if arr[i] > arr[i - j*j] + 1:
      arr[i] = arr[i - j*j] + 1
print(arr[N])