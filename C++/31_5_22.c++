#include <iostream>
#include <stdlib.h>

using namespace std;

struct Node {
	struct Node* next;
	int data;
};

int main()
{
	struct Node* head = (Node*)malloc(sizeof(struct Node));
	struct Node* node1 = (Node*)malloc(sizeof(struct Node));
	head->next = node1;
	node1->data = 10;

	struct Node* node2 = (Node*)malloc(sizeof(struct Node));
	node1->next = node2;
	node2->data = 20;
	node2->next = NULL;

	struct Node* node3 = (Node*)malloc(sizeof(struct Node));
	node3->data = 30;
	node3->next = node2;
	node1->next = node3;

	struct Node* curr = head->next;
	while (curr != NULL) {
		cout << curr->data << endl;
		curr = curr->next;
	}

	free(node1);
	free(node2);
	free(node3);
	free(head);

	return 0;
}

// not working
#include <iostream>
# define MAX_SIZE 8
using namespace std;

int sorted[MAX_SIZE];

int* merge_sort(int arr[]){

  int len = sizeof(&arr);
  if(len<2)
    return arr;

  int mid = len / 2;
  int l = 0;
  int h = 0;
  int idx = 0;
  int low_arr[mid];
  int high_arr[len-mid];

  for(int i=0;i<mid;i++){
    low_arr[i] = arr[i];
  }
  
  for(int i=0;i<mid;i++){
    low_arr[i] = merge_sort(low_arr)[i];
  }
  
  for(int i=mid;i<len;i++){
    high_arr[i-mid] = arr[i];
  }
  
  for(int i=0;i<mid;i++){
    high_arr[i] = merge_sort(high_arr)[i];
  }

  while(l<mid && h<len-mid){
    if(low_arr[l] < high_arr[h]){
      sorted[idx] = low_arr[l];
      l++;
      idx++;
    }
    else{
      sorted[idx] = high_arr[h];
      h++;
      idx++;
    }
  }
  while(idx<len){
    if(l<mid){
      sorted[idx] = low_arr[l];
      l++;
      idx++;
    }
    else if(h<len-mid){
      sorted[idx] = high_arr[h];
      h++;
      idx++;
    }
  }
  return sorted;
}

int main() {
  int arr[8] = {6, 8, 5, 3, 9, 6, 2, 1};
  for(int i=0;i<8;i++){
    cout << merge_sort(arr)[i] << ' ';
  }
  return 0;
}