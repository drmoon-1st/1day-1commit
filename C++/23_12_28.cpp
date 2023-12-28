#include <iostream>
#include <cstring>

using namespace std;

class Node{
    public:
    char name[10]; int num; Node* next;
};

void printList(Node* head){
    Node* cursor = new Node();
    cursor = head->next;
    while(cursor != NULL){
        cout << cursor->num << " " << cursor->name << endl;
        cursor = cursor->next;
    }
}

Node* insert(Node* head, int data, char name[10]){
    Node* newNode = new Node();
    newNode->num = data;
    strcpy(newNode->name, name);
    newNode->next = head->next;
    head->next = newNode->next;
    return head;
}

Node* deleteNode(){

}


int main(){


    return 0;
}