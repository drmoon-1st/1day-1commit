#include <iostream>

using namespace std;

int main()
{
	int arr[3] = { 1,2,3 };
	double arr2[5];
	arr2[0] = 1.333;
	arr2[1] = 2.332322;
	char arr3[2];
	arr3[0] = 'k';
	int temp_arr[5];
	cout << "Enter var to put : " << endl;
	for (int i = 0; i < 5; i++)
	{
		cin >> temp_arr[i];
	}

	int arr[5][2];
	arr[4][1] = 3;
	cout << arr[4][1];

	int var = 5;
	cout << var << " " << &var << " " << var << " " << endl;

	int size;
	int *ptr;

	cout << "Enter size : " << endl;
	cin >> size;
	ptr = new int[size];
	cout << "Size" << endl;
	for (int i = 0; i < size; i++)
	{
		cin >> ptr[i];
	}
	for (int i = 0; i < size; i++)
	{
		cout << ptr[i] << endl;
	}

	int a, b;
	cout << "Enter two numbers" << endl;
	cin >> a >> b;
	cout << max(a, b);



	return 0;
}

int max(int x, int y)
{
	if (x > y)
		return x;
	else
		return y;
}
