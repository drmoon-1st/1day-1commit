import sys
n, r, c = map(int, sys.stdin.readline().split())
target = 0

while True:
    h = 2**(2*(n-1))
    if r >=2**(n-1):
        r -= 2**(n-1)
        target += 2*h
    if c >=2**(n-1):
        c -= 2**(n-1)
        target += h
    n -= 1
    if r == 0 and c == 0:
        break
print(target)

import sys
n = int(sys.stdin.readline())
rem_cnt = sys.stdin.readline()
rem = list(map(int, sys.stdin.readline().split()))
arr = [x for x in range(10) if x not in rem]
a = b = 0
n2 = n - 1
if int(rem_cnt) == 0:
    pass
while True:
    flag = 1
    n2 += 1
    for i in str(n2):
        if int(i) not in arr:
            flag = 0
            break
    if flag == 1:
        a = len(str(n2)) + (n2-n)
        break
n2 = n
while True:
    flag = 1
    n2 -= 1
    for i in str(n2):
        if int(i) not in arr:
            flag = 0
            break
    if flag == 1 or n2 < 0:
        b = len(str(n2)) + (n-n2)
        break
print(min(a, b))

import sys
from collections import deque
def bfs(graph, root):
    visited = []
    queue = deque([root])
    cnt = 0
    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            if n in graph:
                tmp = list(set(graph[n])-set(visited))
                tmp.sort()
                queue += tmp
        cnt += 1
    return cnt

graph = {}
n, m = map(int, sys.stdin.readline().split())
for i in range(m):
    n1, n2 = map(int, sys.stdin.readline().split())
    if n1 not in graph:
        graph[n1] = [n2]
    elif n2 not in graph:
        graph[n2] = [n1]
    
    if n2 not in graph:
        graph[n2] = [n1]
    elif n1 not in graph:
        graph[n1] = [n2]
ans = []
for i in graph:
    ans.append(bfs(graph, i))
print(min(ans))

import sys
from math import inf
n = int(sys.stdin.readline())
m = int(sys.stdin.readline())

graph = [[inf]*n for x in range(n)]
for i in range(m):
    a, b, c = map(int, sys.stdin.readline().split())
    graph[a-1][b-1] = min(graph[a-1][b-1], c)

for i in range(n):
    graph[i][i] = 0
    for j in range(n):
        for k in range(n):
            graph[j][k] = min(graph[j][k], graph[j][i]+graph[i][k])

for i in graph:
    for j in i:
        if j == inf:
            j = 0
        print(j, end=" ")
    print()

import sys

a, b = map(int, sys.stdin.readline().split())
graph = [[a]*a for x in range(a)]
for i in range(b):
    n1, n2 = map(int, sys.stdin.readline().split())
    graph[n1-1][n2-1] = min(graph[n1-1][n2-1], 1)
    graph[n2-1][n1-1] = min(graph[n2-1][n1-1], 1)
for k in range(a):
    graph[k][k] = 0
for k in range(a):
    for i in range(a):
        for j in range(a):
            graph[i][j] = min(graph[i][j], graph[i][k]+graph[k][j])
minarr = []
for i in graph:
    if sum(i) not in graph:
        minarr.append(sum(i))
    else:
        minarr.append(sum(i)+1)
print(minarr.index(min(minarr))+1)