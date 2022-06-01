'''merge sort'''
import sys
a = [3, 4, 5, 7, 2, 9, 8, 0, 10, 0]


def mergesort(arr):
    if len(arr) < 2:
        return arr

    mid = len(arr)//2
    low_arr = mergesort(arr[:mid])
    high_arr = mergesort(arr[mid:])

    merged_arr = []

    l = h = 0
    while l < len(low_arr) and h < len(high_arr):
        if low_arr[l] < high_arr[h]:
            merged_arr.append(low_arr[l])
            l += 1
        else:
            merged_arr.append(high_arr[h])
            h += 1
    merged_arr += low_arr[l:]
    merged_arr += high_arr[h:]
    return merged_arr


print(mergesort(a))

#####


def find(val):
    if val < 2:
        return 0

    for i in range(2, int(val**0.5)+1):
        if val % i == 0:
            return 0

    return 1


a = int(sys.stdin.readline())
m = list(map(int, sys.stdin.readline().split()))
cnt = 0
for i in range(a):
    if find(m[i]) == 1:
        cnt += 1
print(cnt)


class mystack:
    def __init__(self) -> None:
        self.cnt = 0
        self.rear = 0
        self.stk = []

    def push(self, val):
        self.stk.append(val)
        self.cnt += 1
        self.rear += 1

    def pop(self):
        if self.cnt == 0:
            return -1
        t = self.stk[-1]
        self.cnt -= 1
        self.stk.pop()
        return t

    def size(self):
        return self.cnt

    def empty(self):
        if self.cnt == 0:
            return 1
        else:
            return 0

    def top(self):
        if self.cnt == 0:
            return -1
        return self.stk[-1]


s = mystack()
for i in range(int(sys.stdin.readline())):
    a = input()
    if 'push' in a:
        b, c = a.split()
        s.push(int(c))
    elif a == 'top':
        print(s.top())
    elif a == 'size':
        print(s.size())
    elif a == 'empty':
        print(s.empty())
    elif a == 'pop':
        print(s.pop())