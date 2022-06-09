import sys
from collections import deque
def bfs(graph):
    visited = []
    queue = deque([1])
    cnt = 0

    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            cnt += 1
            queue += set(graph[n]) - set(visited)
    return cnt-1

a = int(sys.stdin.readline())
b = int(sys.stdin.readline())
graph = {}
for i in range(b):
    n1, n2 = map(int, sys.stdin.readline().split())

    if n1 not in graph:
        graph[n1] = [n2]
    else:
        graph[n1].append(n2)

    if n2 not in graph:
        graph[n2] = [n1]
    else:
        graph[n2].append(n1)
print(bfs(graph))

import sys
def get(val):
    lst = [1,1,1]
    if val < 3:
       return lst[val-1]
    else:
        for i in range(val-3):
            lst.append(lst[i]+lst[i+1])
    return lst[val-1]
for i in range(int(sys.stdin.readline())):
    print(get(int(sys.stdin.readline())))

import sys
a, b = map(int, sys.stdin.readline().split())
arr = list(map(int, sys.stdin.readline().split()))
s = 0
sarr = [0]
for i in arr:
    s += i
    sarr.append(s)
for i in range(b):
    n1, n2 = map(int, sys.stdin.readline().split())
    print(sarr[n2]-sarr[n1-1])

import sys
def fibo(n):
    cnt = [0, 1]
    if n == 0:
        return cnt[::-1]
    elif n == 1:
        return cnt
    else:
        for i in range(2, n+1):
            cnt.append(cnt[i-1] + cnt[i-2])
        return cnt[i]
a = int(sys.stdin.readline())
print(fibo(a+1)%10007)

import sys
n = int(sys.stdin.readline())
dp = [0]*(n+1)
for i in range(2, n+1):
    dp[i] = dp[i-1] + 1
    if i%2 == 0:
        dp[i] = min(dp[i], dp[i//2]+1)
    if i%3 == 0:
        dp[i] = min(dp[i], dp[i//3]+1)
print(dp[n])

import sys
def fibo(n):
    cnt = [1, 3]
    if n == 0:
        return cnt[0]
    elif n == 1:
        return cnt[1]
    else:
        for i in range(2, n+1):
            cnt.append(cnt[i-1] + 2*cnt[i-2])
        return cnt[i]
a = int(sys.stdin.readline())
print(fibo(a-1)%10007)

import sys
from collections import deque

def bfs(graph, queue):
    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0<=nx<len(graph) and 0<=ny<len(graph[0]) and graph[nx][ny] == 0:
                graph[nx][ny] = graph[x][y] + 1
                queue.append([nx, ny])

a, b = map(int, sys.stdin.readline().split())
graph = []
ans = 0
queue = deque([])
for i in range(b):
    graph.append(list(map(int, sys.stdin.readline().split())))
for i in range(b):
    for j in range(a):
        if graph[i][j] == 1:
            queue.append([i, j])
bfs(graph, queue)
for i in graph:
    for j in i:
        if j == 0:
            print(-1)
            exit(0)
    ans = max(ans, max(i))
print(ans-1)

import sys
a = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
for i in range(1,a):
    arr[i] = max(arr[i], arr[i]+arr[i-1])
print(max(arr))

import sys

def merge_sort(arr):
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
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    return merged_arr

arr = []
for i in range(int(sys.stdin.readline())):
    arr.append(int(sys.stdin.readline()))
for i in merge_sort(arr):
    print(i)

import sys
a = list(sys.stdin.readline().rstrip())
a.sort(reverse=True)
p = ''
for i in a:
    p += i
print(p)

import sys
arr = []
for i in range(int(sys.stdin.readline())):
    x, y = map(int, sys.stdin.readline().split())
    arr.append([x, y])
arr.sort()
for i in arr:
    for j in i:
        print(j, end=' ')
    print()

import sys
arr = []
for i in range(int(sys.stdin.readline())):
    x, y = map(int, sys.stdin.readline().split())
    arr.append([y, x])
arr.sort()
for i in arr:
    for j in reversed(i):
        print(j, end=' ')
    print()

import sys
a = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
ans = [1] * a
for i in range(a):
    for j in range(i):
        if arr[i] > arr[j]:
            ans[i] = max(ans[i], ans[j]+1)
print(max(ans))
p = max(ans)
arr2 = []
for i in range(a-1, -1, -1):
    if ans[i] == p:
        arr2.append(arr[i])
        p -= 1
arr2.reverse()
print(*arr2)

import sys
from bisect import bisect_left
a = int(sys.stdin.readline())
arr = [0]+list(map(int, sys.stdin.readline().split()))
ans = [0] * (a+1)
b = [-1000000001]
maxval = 0
for i in range(1, a+1):
    if b[-1] < arr[i]:
        b.append(arr[i])
        ans[i] = len(b)-1
        maxval = ans[i]
    else:
        ans[i] = bisect_left(b, arr[i])
        b[ans[i]] = arr[i]
p = max(ans)
print(p)
arr2 = []
for i in range(a, 0, -1):
    if ans[i] == p:
        arr2.append(arr[i])
        p -= 1
arr2.reverse()
print(*arr2)