from collections import deque

def bfs(graph, root):
    visited = []
    queue = deque([root])

    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            queue += queue[n] - set(visited)
    return visited

def dfs(graph, root):
    visited = []
    stack = [root]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            stack += graph[n] - set(visited)
    return visited

import sys
from collections import deque

def bfs(graph, root):
    visited = []
    queue = deque([root])

    while queue:
        n = queue.popleft()
        if n not in visited:
            visited.append(n)
            if n in graph:
                tmp = list(set(graph[n])-set(visited))
                tmp.sort()
                queue += tmp
    return ' '.join(str(i) for i in visited)

def dfs(graph, root):
    visited = []
    stack = [root]

    while stack:
        n = stack.pop()
        if n not in visited:
            visited.append(n)
            if n in graph:
                tmp = list(set(graph[n])-set(visited))
                tmp.sort(reverse=True)
                stack += tmp
    return ' '.join(str(i) for i in visited)

n, m, v = map(int, sys.stdin.readline().split())

graph = {}

for i in range(m):
    n1, n2 = map(int, sys.stdin.readline().split())
    if n1 not in graph:
        graph[n1] = [n2]
    elif n2 not in graph[n1]:
        graph[n1].append(n2)
    
    if n2 not in graph:
        graph[n2] = [n1]
    elif n1 not in graph[n2]:
        graph[n2].append(n1)

print(dfs(graph, v))
print(bfs(graph, v))

import sys
from collections import deque

def bfs(graph, b, a):
    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    queue = deque()
    queue.append([b, a])
    graph[b][a] = 0

    while queue:
        y, x = queue.popleft()
        for i in range(4):
            x2 = x + dx[i]
            y2 = y + dy[i]
            if x2 >= len(graph[0]) or y2 >= len(graph) or x2 < 0 or y2 < 0:
                continue
            if graph[y2][x2] == 1:
                graph[y2][x2] = 0
                queue.append([y2, x2])

for i in range(int(sys.stdin.readline())):
    cnt = 0
    m, n, k = map(int, sys.stdin.readline().split())
    graph = [[0]*m for x in range(n)]
    for j in range(k):
        n1, n2 = map(int, sys.stdin.readline().split())
        graph[n2][n1] = 1

    for y in range(n):
        for x in range(m):
            if graph[y][x] == 1:
                bfs(graph, y, x)
                cnt += 1
    print(cnt)

import sys
n, r, c = map(int, sys.stdin.readline().split())
r -= 1
c -= 1
graph = [[0]*(2**n) for _ in range(2**n)]
def z(graph,cnt = 0):
    if len(graph[0]) <= 2:
        graph[0][0] = 0 + cnt
        graph[0][1] = 1 + cnt
        graph[1][0] = 2 + cnt
        graph[1][1] = 3 + cnt
        return graph
    
    lu = []
    ld = []
    ru = []
    rd = []
    for i in range(len(graph[0])//2):
        lu.append(graph[i][:len(graph[0])//2])
        ld.append(graph[len(graph[0])//2+i][:len(graph[0])//2])
        ru.append(graph[i][len(graph[0])//2:])
        rd.append(graph[len(graph[0])//2+i][len(graph[0])//2:])    
    
    p = z(lu, cnt)
    cnt += 4
    q = z(ru, cnt)
    cnt += 4
    r = z(ld, cnt)
    cnt += 4
    s = z(rd, cnt)
    for i in range(len(graph[0])//2):
        graph[i][:len(graph[0])//2] = p[i]
        
        graph[len(graph[0])//2+i][:len(graph[0])//2] = r[i]
        
        graph[i][len(graph[0])//2:] = q[i]
        
        graph[len(graph[0])//2+i][len(graph[0])//2:] = s[i]

    print(p)
    print(q)
    print(r)
    print(s)
print(z(graph))