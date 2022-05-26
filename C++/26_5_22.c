#include <stdio.h>


//'''bubble sort'''
int main(void) {
  int arr[8] = {1, 3, 5, 7, 2, 4, 8, 6};
  
  for(int i=0;i<sizeof(arr)/sizeof(int)-1;i++)
    {
      short Swap = 0;
      int tmp;
      for(int j=0;j<sizeof(arr)/sizeof(int)-i-1;j++)
        {
          if(arr[j] > arr[j+1])
          {
            tmp = arr[j+1];
            arr[j+1] = arr[j];
            arr[j] = tmp;
            Swap = 1;
          }
        }
      if(Swap == 0)
        break;
    }

  for(int i=0;i<8;i++)
    {
      printf("%d\t", arr[i]);
    }
  return 0;
}