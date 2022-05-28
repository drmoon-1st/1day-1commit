#include <iostream>
using namespace std;
#define MAX 100

int front = 0;
int rear = 0;

int que[MAX];

int IsEmpty()
{
    if (rear == front)
        return 1;
    return 0;
}

int IsFull()
{
    if ((rear + 1) % MAX == front)
        return 1;
    return 0;
}

void enqueue(int value)
{
    if (IsFull())
        cout << "FULL" << endl;
    else
    {
        rear = (rear + 1) % MAX;
        que[rear] = value;
    }
}

int dequeue()
{
    if (IsEmpty())
    {
        cout << "EMPTY" << endl;
        return -1;
    }
    else
    {
        front = (front + 1) % MAX;
        return que[front];
    }
}

int main()
{
    cout << IsEmpty() << endl;
    cout << IsFull() << endl;
    enqueue(10);
    return 0;
}

#include <iostream>
using namespace std;

int main()
{
    int arr[9] = {4, 2, 6, 8, 9, 3, 1, 2, 0};
    int n = sizeof(arr) / sizeof(int);
    int h = n / 2;

    while (h > 0)
    {
        for (int i = h; i < n; i++)
        {
            int j = i - h;
            int tmp = arr[i];
            while (j >= 0 && arr[j] > tmp)
            {
                arr[j + h] = arr[j];
                j = j - h;
            }
            arr[j + h] = tmp;
        }
        h = h / 2;
    }

    for (int i = 0; i < 9; i++)
    {
        cout << arr[i] << ' ';
    }

    return 0;
}

#include <iostream>
using namespace std;

int main()
{
    int arr[8] = {3, 5, 7, 8, 1, 2, 9, 0};

    for (int i = 0; i < sizeof(arr) / sizeof(int); i++)
    {
        int min = i;
        int tmp;
        for (int j = i + 1; j < sizeof(arr) / sizeof(int); j++)
        {
            if (arr[min] > arr[j])
                min = j;
        }
        tmp = arr[min];
        arr[min] = arr[i];
        arr[i] = tmp;
    }

    for (int i = 0; i < 8; i++)
    {
        cout << arr[i] << ' ';
    }
    return 0;
}