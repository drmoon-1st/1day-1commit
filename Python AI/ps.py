import sys
from itertools import combinations_with_replacement

input = sys.stdin.readline
n, m = map(int, input().split(' '))
arr = sorted(list(map(int, input().split(' '))))
for i in sorted(set(combinations_with_replacement(arr, m))):
    print(*i)
