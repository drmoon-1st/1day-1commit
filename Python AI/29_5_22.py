'''이분 탐색'''
import sys
arr = [3, 6, 5, 8, 7, 9, 0, 1, 2]
tgt = [3, 2, 1]
arr = sorted(arr)
for i in tgt:
    front = 0
    rear = len(arr)-1
    while True:
        mid = (front + rear) // 2
        if i == arr[mid]:
            break
        elif i < arr[mid]:
            rear = mid - 1
        else:
            front = mid + 1
    print(arr[mid])
print('end')

'''bubble sort'''
arr = [3, 6, 5, 8, 7, 9, 0, 1, 2]
for i in range(len(arr)-1):
    swap = 0
    for j in range(len(arr)-1-i):
        if arr[j] > arr[j+1]:
            arr[j], arr[j+1] = arr[j+1], arr[j]
            swap = 1
    if swap == 0:
        break
print(arr)

'''insert sort'''
arr = [3, 6, 5, 8, 7, 9, 0, 1, 2]
for i in range(len(arr)-1):
    for j in range(i+1, 0, -1):
        if arr[j] < arr[j-1]:
            arr[j], arr[j-1] = arr[j-1], arr[j]
print(arr)

'''shell sort'''
arr = [3, 6, 5, 8, 7, 9, 0, 1, 2]
n = len(arr)
h = n//2
while h > 0:
    for i in range(h, n):
        j = i - h
        tgt = arr[i]  # arr[h] for first
        while j >= 0 and arr[j] > tgt:
            arr[j+h] = arr[j]
            j -= h
        arr[j+h] = tgt
    h //= 2
print(arr)

'''selection sort'''
arr = [3, 6, 5, 8, 7, 9, 0, 1, 2]
for i in range(len(arr)-1):
    min = i
    for j in range(i+1, len(arr)):
        if arr[min] > arr[j]:
            min = j
    arr[min], arr[i] = arr[i], arr[min]
print(arr)

n, m = map(int, sys.stdin.readline().split())
a1 = [int(sys.stdin.readline()) for _ in range(n)]
a2 = [int(sys.stdin.readline()) for _ in range(m)]
end = sys.stdin.readline()
cnt = 0

for i in a1:
    front = 0
    rear = m
    while front <= rear:
        mid = (front + rear) // 2
        if a2[mid] == i:
            cnt += 1
            break
        elif a2[mid] > i:
            rear = mid - 1
        else:
            front = mid + 1
print(cnt)