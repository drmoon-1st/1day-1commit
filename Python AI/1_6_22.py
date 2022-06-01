import sys
n = int(sys.stdin.readline())

que=[]
for i in range(n):
    command = sys.stdin.readline().split()

    if command[0]=='push':
        que.append(command[1])
    elif command[0]=='pop':
        if len(que)==0:
            print(-1)
        else:
            print(que[0])
            for i in range(1, len(que)):
                que[i-1] = que[i]
            que.pop()
    elif command[0] == 'size':
        print(len(que))
    elif command[0] == 'empty':
        if len(que)==0:
            print(1)
        else:
            print(0)
    elif command[0] == 'front':
        if len(que)==0:
            print(-1)
        else:
            print(que[0])
    elif command[0] == 'back':
        if len(que)==0:
            print(-1)
        else:
            print(que[-1])

import sys

for _ in range(int(sys.stdin.readline())):
    arr = list(map(str, sys.stdin.readline()))
    p = 0
    for i in arr:
        if p == 0 and i == ')':
            p = 1
            break
        if i == '(':
            p += 1
        else:
            p -= 1
    p += 1
    if p == 0:
        print("YES")
    else:
        print("NO")

a,b = map(int, input().split())
print(a+b)

from bisect import insort


class Node:
    def __init__(self, data) -> None:
        self.left = None
        self.right = None
        self.data = data 

n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)
n6 = Node(6)
n7 = Node(7)
n8 = Node(8)

class binary_tree:
    def __init__(self) -> None:
        self.root = None

    def preorder(self, n):
        if n != None:
            print(n.data, end='')
            if n.left:
                self.preorder(n.left)
            if n.right:
                self.preorder(n.right)

    def postorder(self, n):
        if n != None:
            if n.left:
                self.postorder(n.left)
            if n.right:
                self.postorder(n.right)
            print(n.data, end='')

    def inorder(self, n):
        if n!= None:
            if n.left:
                self.inorder(n.left)
            print(n.data, end='')
            if n.right:
                self.inorder(n.right)

    def levelorder(self, n):
        q = []
        q.append(n)
        while q:
            t = q.pop(0)
            print(t.data, end='')
            if t.left:
                q.append(t.left)
            if t.right:
                q.append(t.right)


tree = binary_tree()
tree.root = n1
n1.left = n2
n1.right = n3
n2.left = n4
n2.right = n5
n3.left = n6
n3.right = n7
n4.left = n8

tree.preorder(n1)
tree.postorder(n1)
tree.inorder(n1)
tree.levelorder(n1)

import sys
import math

for _ in range(int(sys.stdin.readline())):
    arr = list(map(int, sys.stdin.readline().split()))
    dist = math.sqrt((arr[0]-arr[3])**2+(arr[1]-arr[4])**2)
    if dist == 0:
        if arr[2] == arr[5]:
            print(-1)
        else:
            print(0)
    else:
        if dist > arr[2] + arr[5]:
            print(0)
        elif dist+arr[2] == arr[5] or dist+arr[5] == arr[2]:
            print(1)
        elif dist < arr[2] + arr[5] and dist+arr[2] < arr[5] and dist+arr[5] < arr[2]:
            print(2)
        else:
            print(0)

#########

from constantly import ValueConstant
import nltk
from nltk.corpus import names
print(names.words())
m = names.words('male.txt')
f = names.words('female.txt')

def mk_fest(d):
    return {'last_char': d[-1], 'first_char': d[0]}

names_labelid = [(mk_fest(x), 'male') for x in m] +[(mk_fest(x), 'demale') for x in f]
nb = nltk.NaiveBayesClassifier.train(names_labelid)
nb.classify({'last_char': 'x', 'first_char': 'B'})
print(nltk.classify.accuracy(nb, names_labelid[:500]))

data = []
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(data).toarray())
print(vec.get_feature_names())

from sklearn.feature_extraction.text import CountVectorizer
txt = ['this is first line.', 'this is second line.']

cv = CountVectorizer()
cv.fit(txt)
print(cv.get_feature_names())
print(cv.transform(['this is line line.']).toarray())

import pandas as pd

pd_data = pd.read_csv('labeledTrainData.tsv', sep='\t')
print(pd_data)
from bs4 import BeautifulSoup

with_tag = pd_data['review'][0]
soup = BeautifulSoup(with_tag, 'html.parser').get_text()
print(soup)

import nltk
nltk.download('stopwords')
token = nltk.word_tokenize(soup)
from nltk.corpus import stopwords
print(stopwords.words('english'))
stopword_filter = [w for w in token if w not in stopwords.words('english')]
non_alpha = [w for w in stopword_filter if w.isalpha()]

data = pd.DataFrame(non_alpha)
print(data.value_counts().head(10))
'MJ, people, one, know, movie, sequence, going, like, maybe, message'
print(data.value_counts())