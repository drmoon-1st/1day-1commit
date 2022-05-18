#include <iostream>

using namespace std;

int main()
{
	int x;
	x = 5;
	int y = x + 5;
	int a, b;
	a = 5;
	b = 3;
	cout << "the value of variable x is :" << x;
	cout << "the value of variable y is :" << y;
	cout << "the value of variable a is :" << a;
	cout << "the value of variable b is :" << b;

	int a, b;
	a = 5;
	cout << ++a;
	while (true)
	{
		cout << "Plesae enter a number : ";
		int x;
		cin >> x;
		if (x >= 0)
		{
			cout << "pos" << endl;
		}
		else
		{
			cout << "neg" << endl;
		}
		cout << "restart? : ";
		string yes;
		cin >> yes;
		if (yes == "n")
		{
			break;
		}
	}



	return 0;
}