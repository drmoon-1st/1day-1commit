import sys
import math
a, b = map(int, sys.stdin.readline().split())

arr = [True for _ in range(a, b+1)]
arr2 = [x**2 for x in range(2, int((b)**0.5)+1)]
answer = 0

for i in arr2:
    idx = (math.ceil(a/i)*i)-a
    while True:
        arr[idx] = False
        idx += i
        if idx <= (b-a):break

for i in range(len(arr)):
    if arr[i] ==  True:answer+=1
print(answer)