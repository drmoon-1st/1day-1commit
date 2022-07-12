import sys

n = int(sys.stdin.readline())

data = [] # ti, pi 저장
arr = [0 for _ in range(n+1)] # 최고값 저장

for i in range(n):
    t, p = map(int, sys.stdin.readline().split())
    data.append([t, p])

# for i in range(n-1, -1, -1):
#     if data[i][0] + i > n:
#         arr[i] = arr[i+1]
#     else:
#         arr[i] = max(arr[i+1], arr[i + data[i][0]] + data[i][1])

# print(arr[0])

for i in range(1, n+1):
    if data[i-1][0] + i > n:
        arr[i] = arr[i-1]
    else:
        arr[i] = max(arr[i-1], arr[i + data[i-1][0]] + data[i-1][1])

print(arr)