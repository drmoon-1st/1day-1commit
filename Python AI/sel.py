import sys
from itertools import permutations

a, b = map(int, sys.stdin.readline().split(' '))
comb = permutations(sorted(list(map(int, sys.stdin.readline().split(' ')))), b)

for i in comb:
    flag = True
    rear = 0
    for j in i:
        if j < rear:
            flag = False
            break
        else:
            rear = j
    if flag:print(*i)

import sys
from itertools import permutations

a, b = map(int, sys.stdin.readline().split(' '))
comb = permutations(sorted(list(map(int, sys.stdin.readline().split(' ')))), b)

for i in comb:
    print(*i)

import sys
from itertools import combinations_with_replacement

a, b = map(int, sys.stdin.readline().split(' '))
comb = combinations_with_replacement(sorted(list(map(int, sys.stdin.readline().split(' ')))), b)

for i in comb:
    flag = True
    rear = 0
    for j in i:
        if j < rear:
            flag = False
            break
        else:
            rear = j
    if flag:print(*i)