#include <iostream>
using namespace std;
//'''bubble sort'''
int main() {
  int arr[8] = {1, 3, 5, 7, 2, 4, 8, 6};

  for(int i=0;i<sizeof(arr)/sizeof(int)-1;i++)
    {
      int tmp = 0;
      short swap = 0;
      for(int j=0;j<sizeof(arr)/sizeof(int)-i-1;j++)
        {
          if(arr[j] > arr[j+1])
          {
            tmp = arr[j+1];
            arr[j+1] = arr[j];
            arr[j] = tmp;
            swap = 1;
          }
        }
    if(swap == 0)
      break;
    }

  for(int i=0;i<sizeof(arr)/sizeof(int);i++)
    {
      std::cout << arr[i] << ' ';
    }
  return 0;
}

//'''array'''

class Array{

  private:
    int* arr;
    int arrSize;
  public:
    Array(int size){
      arrSize = size;
      arr = new int[arrSize];

      for(int i=0;i<size;i++){
        arr[i] = 0;
      }
    }

    int at(int idx){
      return arr[idx];
    }

    void append(int idx, int value){
      if(idx > arrSize-1)
        cout << "Error" << endl;
      else
        for(int i=arrSize-2;i>=idx;i--){
          arr[i+1] = arr[i];
        }
        arr[idx] = value;
    }

    void remove(int idx){
      for(int i=idx;i<arrSize-1;i++){
        arr[i] = arr[i+1];
      }
      arr[arrSize-1] = 0;
    }

    void set(int idx, int value){
      arr[idx] = value;
    }

    int len(){
      return arrSize;
    }

    void pop(){
      arr[arrSize-1] = 0;
    }

    void reverse(){
      int tmp;
      for(int i=0;i<=arrSize-2;i++){
        tmp = arr[arrSize-1];
        for(int j=arrSize-2;j>=i;j--){
          arr[j+1] = arr[j];
        }
        arr[i] = tmp;
      }
    }

    int count(int idx){
      int cnt = 0;
      for(int i=0;i<arrSize;i++){
        if(arr[i] == idx)
          cnt ++;
      }
      return cnt;
    }

    void display(){
      for(int i=0;i<arrSize;i++){
        cout << arr[i] << ' ';
      }
    }
};

int main() {

  Array a(5);
  a.append(3, 3);
  a.append(2, 4);
  cout << a.count(0) << endl;
  a.pop();
  a.append(4, 2);
  // a.remove(2);
  a.reverse();
  a.display();
  
  return 0;
}