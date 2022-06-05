import sys
a = list(map(int, sys.stdin.readline().split()))
a = sorted(a)
def ans(arr):
    cnt = arr[0]
    cnt2 = arr[0]
    while True:
        if arr[0] % cnt == 0 and arr[1] % cnt == 0:
            break
        cnt -= 1
    p = 1
    while True:
        if (cnt2 * p) % arr[1] == 0:
            break
        p += 1
    print(cnt)
    print(cnt2*p)
ans(a)

import sys
a, b = map(int, sys.stdin.readline().split())
arr = []
p = 1
while b>=p:
    for i in range(p):
        arr.append(p)
    p += 1
print(sum(arr[a-1:b]))

import sys
a = int(sys.stdin.readline())
b = int(sys.stdin.readline())
def isans(val):
    for i in range(2, int(val**0.5)+1):
        if val % i == 0:
            return 0
    return 1
arr = []
for i in range(a, b+1):
    if isans(i) == 1:
        arr.append(i)
if 1 in arr:
    arr.remove(1)
print(-1 if len(arr)==0 else sum(arr))
if len(arr) != 0:
    print(arr[0])

import sys
from itertools import permutations
n = int(sys.stdin.readline())
arr = list(map(int, sys.stdin.readline().split()))
ptr = list(map(int, sys.stdin.readline().split()))
ptr2 = []
k = '+-*/'
for i in range(4):
    for j in range(ptr[i]):
        ptr2.append(k[i])
del ptr

def get_ans(arr, ptr):
    p = permutations(ptr)
    p = list(set(p))
    ans = []
    cnt = arr[0]
    for i in p:
        r = 1
        cnt = arr[0]
        for j in i:
            if j == '+':
                cnt += arr[r]
                r += 1
            elif j == '-':
                cnt -= arr[r]
                r += 1
            elif j == '*':
                cnt *= arr[r]
                r += 1
            else:
                cnt /= arr[r]
                cnt = int(cnt)
                r += 1
        ans.append(cnt)
    return ans
print(max(get_ans(arr, ptr2)))
print(min(get_ans(arr, ptr2)))

import sys
a = list(map(str, sys.stdin.readline()))
def find(arr):
    while True:
        for i in range(len(arr)-1):
            if arr[i] == '[' and arr[i+1] == ']':
                arr[i+1] = 3
                arr.pop(i)
            elif arr[i] == '(' and arr[i+1] == ')':
                pass
        
        for i in range(len(arr)-2):
            if arr[i] == '[' and type(arr[i+1]) == int and arr[i+2] == ']':
                arr[i+1] *= 3
                arr.pop(i)
                arr.pop(i+2)
            elif arr[i] == '(' and type(arr[i+1]) == int and arr[i+2] == ')':
                arr[i+1] *= 2
                arr.pop(i)
                arr.pop(i+2)
        if len(arr) <= 1:
            break
    print(arr)
    return 1
find(a)