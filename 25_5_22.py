from tkinter.tix import Tree


hash_table = list([i for i in range(10)])

def hash_func(key):
    return key % 5

data1 = 'a'
data2 = 'b'
data3 = 'c'
data4 = 'd'

def store(data, value):
    key = ord(data[0])
    hash_adress = hash_func(key)
    hash_table[hash_adress] = value

store(data1, '0101013')
store(data2, '990199913')
store(data3, '01433')

def get_data(data):
    key = ord(data[0])
    hash_adress = hash_func(key)
    return hash_table[hash_adress]

print(get_data(data1))

class Node:
    def __init__(self, value) -> None:
        self.value = value
        self.left = None
        self.right = None

class NodeMgnt:
    def __init__(self, head) -> None:
        self.head = head

    def insert(self, value):
        self.current_node = self.head
        while True:
            if value < self.current_node.value:
                if self.current_node.left != None:
                    self.current_node = self.current_node.left
                else:
                    self.current_node.left = Node(value)
                    break
            else:
                if self.current_node.right != None:
                    self.current_node = self.current_node.right
                else:
                    self.current_node.right = Node(value)
                    break

    def search(self, value):
        self.current_node = self.head
        while self.current_node:
            if self.current_node.value == value:
                return True
            elif value < self.current_node.value:
                self.current_node = self.current_node.left
            else:
                self.current_node = self.current_node.right
        return False

head = Node(1)
bst = NodeMgnt(head)
bst.insert(2)
bst.insert(3)
bst.insert(8)
bst.insert(0)
bst.insert(5)

print(bst.search(8))

class Heap:
    def __init__(self, data) -> None:
        self.heap_array = list()
        self.heap_array.append(None)
        self.heap_array.append(data)

    def move_up(self, inserted_idx):
        if inserted_idx <= 1:
            return False
        parent_idx = inserted_idx // 2
        if self.heap_array[inserted_idx] > self.heap_array[parent_idx]:
            return True
        else:
            return False

    def insert(self, data):
        if len(self.heap_array) == 0:
            self.heap_array.append(None)
            self.heap_array.append(data)
            return True
        self.heap_array.append(data)

        inserted_idx = len(self.heap_array) - 1
        
        while self.move_up(inserted_idx):
            parent_idx = inserted_idx // 2
            self.heap_array[inserted_idx], self.heap_array[parent_idx] = self.heap_array[parent_idx], self.heap_array[inserted_idx]
            inserted_idx = parent_idx
        return True

    def move_down(self, popped_idx):
        left_child_popped_idx = popped_idx * 2
        right_child_popped_idx = popped_idx * 2 + 1

        if left_child_popped_idx >= len(self.heap_array):
            return False

        if right_child_popped_idx >= len(self.heap_array):
            if self.heap_array[popped_idx] < self.heap_array[left_child_popped_idx]:
                return True
            else:
                return False

        
        

    def pop(self):
        if len(self.heap_array) <= 1:
            return None
        returned_data = self.heap_array[1]
        self.heap_array[1] = self.heap_array[-1]
        del self.heap_array[-1]
        popped_idx = 1

        while self.move_down(popped_idx):
            left_child_popped_idx = popped_idx * 2
            right_child_popped_idx = popped_idx * 2 + 1

            if right_child_popped_idx >= len(self.heap_array):
                if self.heap_array[popped_idx] < self.heap_array[left_child_popped_idx]:
                    self.heap_array[popped_idx], self.heap_array[left_child_popped_idx] = self.heap_array[left_child_popped_idx], self.heap_array[popped_idx]




heap = Heap(1)
print(heap.heap_array)

arr = [int(input()) for x in range(int(input()))]
stack_table = []
ans = []
for i in range(len(arr)):
    if len(stack_table) != 0 and arr[i] == stack_table[-1]:
        ans.append('-')
        stack_table.pop(-1)
    else:
        for j in range(1, arr[i]+1):
            if j in arr[i:] and j not in stack_table:
                stack_table.append(j)
                ans.append('+')
        if arr[i] == stack_table[-1]:
            ans.append('-')
            stack_table.pop(-1)
if len(stack_table) != 0:
    print('NO')
else:
    for i in ans:
        print(i)

a, b = map(int, input().split())
card = list(map(int, input().split()))
max = 0
for i in range(len(card)-2):
    for j in range(1, len(card)-1):
        for k in range(2, len(card)):
            if i != j and j != k and k != i:
                p = card[i] + card[j] + card[k]
                if max <= p <= b:
                    max = p
print(max)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import SCORERS, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
X = X[:, (0, 1, 3)]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

dt = DecisionTreeClassifier(max_depth=7).fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_pred_proba = dt.predict_proba(X_test)

print(y_test)
print(y_pred)

data = pd.read_csv("bean.csv", header=0)
X = data[["ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]]
y = data[["Class"]]
y = pd.Series(y.values.squeeze(), dtype='category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

dt = DecisionTreeClassifier(max_depth=7, max_leaf_nodes=15)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(dt.score(X_test, y_test))
f, ax = plt.subplots(1,2, figsize=(17, 10))
ax[0].scatter(np.arange(0, len(y_pred)), y_test, color='red')
ax[1].scatter(np.arange(0, len(y_pred)), y_pred)
plt.show()
p = precision_score(y_test, y_pred, average=None)
print(p)
fpr, tpr, tr = roc_curve(y_test, dt.predict_proba(X_test)[:, 1], pos_label=1)
f, ax = plt.subplots()
ax.plot(fpr, tpr)
plt.show()