import sys
from itertools import combinations, permutations

input = sys.stdin.readline
n = int(input())
graph = [list(map(int, input().split())) for _ in range(n)]

comb = list(combinations([x for x in range(n)], n//2))
front, end = 0, len(comb)-1

answer = inf
while front < end:
    start = link = 0
    for a, b in list(permutations(comb[front], 2)):
        start += graph[a][b]
    for a, b in list(permutations(comb[end], 2)):
        link += graph[a][b]
    answer = min(answer, ((start-link)**2)**0.5)
    front += 1;end -= 1
print(int(answer))