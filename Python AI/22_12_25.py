# import sys
# sys.setrecursionlimit(10**6)
# input = sys.stdin.readline

# def DFS(n:int, tree:list, num:list):
#     for i in tree[n]:
#         if num[i] == -1:
#             num[i] = n;DFS(i, tree, num)

# k = int(input())
# tree = [[] for x in range(k+1)]
# num = [-1] * (k+1)

# for i in range(k-1):
#     a, b = map(int, input().split(' '))
#     tree[a].append(b)
#     tree[b].append(a)

# DFS(1, tree, num)
# p = [print(num[x]) for x in range(2, k+1)]

################

# import sys
# from itertools import permutations
# input = sys.stdin.readline
# n, m = map(int, input().split(' '))
# rear = 'k'
# for i in sorted(set(permutations(list(map(int, input().split(' '))), m))):
#     if i == rear:pass
#     else:rear = i;print(*i[0:m])

################

