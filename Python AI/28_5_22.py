import math
import sys
ans = ''
stk = []
p = 1
for i in range(int(sys.stdin.readline())):
    a = int(sys.stdin.readline())
    while True:
        if p > a:
            break
        stk.append(p)
        ans += '+'
        p += 1
    if stk[-1] == a:
        stk.pop()
        ans += '-'
    else:
        ans = ''
        break
for i in ans:
    print(i)
if ans == '':
    print("NO")

###


class fQue:
    class Empty(Exception):
        pass

    class Full(Exception):
        pass

    def __init__(self, capacity) -> None:
        self.no = 0
        self.front = 0
        self.rear = 0
        self.capacity = capacity
        self.que = [None] * capacity

    def is_empty(self):
        return self.no <= 0

    def is_full(self):
        return self.no >= self.capacity

    def enqueue(self, value):
        if self.is_full():
            raise fQue.Full
        self.que[self.rear] = value
        self.rear += 1
        self.no += 1
        if self.rear == self.capacity:
            self.rear = 0

    def dequeue(self):
        if self.is_empty():
            raise fQue.Empty
        x = self.que[self.front]
        self.front += 1
        self.no -= 1
        if self.front == self.capacity:
            self.front = 0
        return x

    def peek(self, idx):
        return self.que[idx]

    def display(self):
        print(self.que)


a = fQue(3)
a.enqueue(3)
a.enqueue(2)
a.enqueue(5)
print(a.dequeue())
print(a.dequeue())
a.display()

a, b = map(int, sys.stdin.readline().split())


def find(val):
    if val == 1:
        return 0
    p = int(math.sqrt(val))
    for i in range(2, p+1):
        if val % i == 0:
            return 0
    return 1


for i in range(a, b+1):
    if find(i) == 1:
        print(i)


a = [3, 5, 8, 9, 4, 9, 7, 4]


def shell_sort(data):
    n = len(data)
    h = n//2
    while h > 0:
        for i in range(h, n):
            j = i - h
            tmp = data[i]
            while j >= 0 and data[j] > tmp:
                data[j+h] = data[j]
                j -= h
            data[j+h] = tmp
        h //= 2


shell_sort(a)
print(a)


def selection_sort(data):
    for stand in range(len(data)-1):
        lowest = stand
        for index in range(stand+1, len(data)):
            if data[lowest] > data[index]:
                lowest = index
        data[lowest], data[stand] = data[stand], data[lowest]


selection_sort(a)
print(a)


def fact(n):
    if n > 1:
        return n * fact(n-1)
    else:
        return n


def multiply(n):
    if n > 1:
        return n * multiply(n-1)
    else:
        return n


print(multiply(4))