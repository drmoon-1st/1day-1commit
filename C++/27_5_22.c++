#include <iostream>
using namespace std;

class fStack
{
private:
    int *arr;
    int arrSize;
    int ptr = 0;

public:
    fStack(int size)
    {
        arrSize = size;
        arr = new int[arrSize];
        for (int i = 0; i < arrSize; i++)
        {
            arr[i] = 0;
        }
    }

    int is_full()
    {
        return ptr >= arrSize;
    }

    int is_empty()
    {
        return ptr <= 0;
    }

    void push(int value)
    {
        if (is_full())
        {
            cout << "full" << endl;
        }
        for (int i = ptr; i > 0; i--)
        {
            arr[i] = arr[i - 1];
        }
        arr[0] = value;
        ptr++;
    }

    int pop()
    {
        if (is_empty())
        {
            cout << "empty" << endl;
        }
        int ret = arr[0];
        for (int i = 1; i <= ptr; i++)
        {
            arr[i - 1] = arr[i];
        }
        ptr--;
        return ret;
    }

    int peek(int idx)
    {
        return arr[idx];
    }

    void display()
    {
        for (int i = 0; i < arrSize; i++)
        {
            cout << arr[i] << " ";
        }
    }
};

int main()
{
    fStack a(10);
    a.push(4);
    a.push(3);
    a.push(5);
    a.push(7);
    a.pop();
    a.pop();
    a.display();

    return 0;
}

#include <iostream>
using namespace std;

class fStack
{
private:
    int *arr;
    int arrSize;
    int ptr = 0;

public:
    fStack(int size)
    {
        arrSize = size;
        arr = new int[arrSize];
        for (int i = 0; i < arrSize; i++)
        {
            arr[i] = 0;
        }
    }

    int is_full()
    {
        return ptr >= arrSize;
    }

    int is_empty()
    {
        return ptr <= 0;
    }

    void push(int value)
    {
        if (is_full())
        {
            cout << "full" << endl;
        }
        arr[ptr] = value;
        ptr++;
    }

    int pop()
    {
        if (is_empty())
        {
            cout << "empty" << endl;
        }
        int ret = arr[0];
        for (int i = 1; i <= ptr; i++)
        {
            arr[i - 1] = arr[i];
        }
        ptr--;
        return ret;
    }

    int peek(int idx)
    {
        return arr[idx];
    }

    void display()
    {
        for (int i = 0; i < arrSize; i++)
        {
            cout << arr[i] << " ";
        }
    }
};

int main()
{
    fStack a(10);
    a.push(4);
    a.push(3);
    a.push(5);
    a.push(7);
    a.pop();
    a.pop();
    a.display();

    return 0;
}

#include <iostream>
using namespace std;

int main()
{
    int arr[6] = {2, 5, 3, 7, 1, 4};
    for (int i = 0; i < sizeof(arr) / sizeof(int) - 1; i++)
    {
        int swap = 0;
        int tmp;
        for (int j = 0; j < sizeof(arr) / sizeof(int) - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                tmp = arr[j + 1];
                arr[j + 1] = arr[j];
                arr[j] = tmp;
                swap = 1;
            }
        }
        if (swap == 0)
            break;
    }
    for (int i = 0; i < 6; i++)
    {
        cout << arr[i] << ' ';
    }

    return 0;
}

#include <iostream>
using namespace std;

int main()
{
    int arr[6] = {2, 5, 3, 7, 1, 4};
    for (int i = 0; i < sizeof(arr) / sizeof(int) - 1; i++)
    {
        int tmp;
        for (int j = i + 1; j >= 1; j--)
        {
            if (arr[j - 1] > arr[j])
            {
                tmp = arr[j];
                arr[j] = arr[j - 1];
                arr[j - 1] = tmp;
            }
            else
            {
                break;
            }
        }
    }
    for (int i = 0; i < 6; i++)
    {
        cout << arr[i] << ' ';
    }
    return 0;
}