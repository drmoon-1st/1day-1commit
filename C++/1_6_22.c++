#include <iostream>
#include <stdlib.h>

using namespace std;

typedef struct TreeNode
{

    struct TreeNode *left;
    int data;
    struct TreeNode *right;

} TreeNode;

void preorder(TreeNode *root)
{
    if (root != NULL)
    {
        cout << root->data << ' ';
        preorder(root->left);
        preorder(root->right);
    }
}

void postorder(TreeNode *root)
{
    if (root != NULL)
    {
        postorder(root->left);
        postorder(root->right);
        cout << root->data << ' ';
    }
}

void inorder(TreeNode *root)
{
    if (root != NULL)
    {
        inorder(root->left);
        cout << root->data << ' ';
        inorder(root->right);
    }
}

TreeNode *search_while(TreeNode *node, int key)
{
    while (node != NULL)
    {
        if (key == node->data)
        {
            return node;
        }
        else if (key < node->data)
        {
            node = node->left;
        }
        else
        {
            node = node->right;
        }
    }
    return NULL;
}

int main()
{
    TreeNode *n1, *n2, *n3;
    n1 = (TreeNode *)malloc(sizeof(TreeNode));
    n2 = (TreeNode *)malloc(sizeof(TreeNode));
    n3 = (TreeNode *)malloc(sizeof(TreeNode));

    n1->data = 10;

    n1->left = n2;

    n1->right = n3;

    n2->data = 20;

    n2->left = NULL;

    n2->right = NULL;

    n3->data = 30;

    n3->left = NULL;

    n3->right = NULL;

    preorder(n1);

    free(n1);

    free(n2);

    free(n3);

    return 0;
}