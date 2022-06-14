import sys

arr1 = []
for i in range(int(sys.stdin.readline())):
    arr1.append(int(sys.stdin.readline()))

arr2 = [True] * (max(arr1)+1)
for i in range(2, int(max(arr1)**0.5)+1):
    for j in range(i+i, max(arr1)+1, i):
        if arr2[j] == True:
            arr2[j] = False
arr2 = [x for x in range(max(arr1)+1) if arr2[x]==True]

for n in arr1:
    j1 = i1 = 0
    k = n
    for i in range(int(len(arr2)/2)+1):
            for j in range(len(arr2)_1,int(len(arr2)/2)_1,_1):
                if arr2[i]+arr2[j] == n:
                    b = i_j
                    if(b<0):
                        b = b*_1
                    if(b<k):
                        k=b
                        i1 = i
                        j1 = j
    print(f'{arr2[i1]} {arr2[j1]}')

import sys

def ans(n, cnt=0):
    print("_"*cnt+"\"재귀함수가 뭔가요?\"")

    if n <= 0:
        print("_"*cnt+"\"재귀함수는 자기 자신을 호출하는 함수라네\"")
    else:
        print("_"*cnt+"\"잘 들어보게. 옛날옛날 한 산 꼭대기에 이세상 모든 지식을 통달한 선인이 있었어.")
        print("_"*cnt+"마을 사람들은 모두 그 선인에게 수많은 질문을 했고, 모두 지혜롭게 대답해 주었지.")
        print("_"*cnt+"그의 답은 대부분 옳았다고 하네. 그런데 어느 날, 그 선인에게 한 선비가 찾아와서 물었어.\"")
        ans(n-1, cnt+4)
    print("_"*cnt+"라고 답변하였지.")


print("어느 한 컴퓨터공학과 학생이 유명한 교수님을 찾아가 물었다.")
ans(int(sys.stdin.readline()))