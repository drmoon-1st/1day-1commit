#include <stdio.h>

int main(void)
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
        {
            break;
        }
    }
    for (int i = 0; i < 6; i++)
    {
        printf("%d\t", arr[i]);
    }

    return 0;
}

#include <stdio.h>

int main(void)
{
    int arr[6] = {2, 5, 3, 7, 1, 4};
    for (int i = 0; i < sizeof(arr) / sizeof(int) - 1; i++)
    {
        int tmp;
        for (int j = i + 1; j >= 1; j--)
        {
            if (arr[j] < arr[j - 1])
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
        printf("%d\t", arr[i]);
    }
    return 0;
}