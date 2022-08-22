#include <iostream>

using namespace std;

int main()
{
	int arr[4] = { 1, 4, 3, 2 };
	int tmp;

	for (int i = 0; i < 3; i++) {
		int j = i;
		while (arr[j] > arr[j + 1]) {
			tmp = arr[j];
			arr[j] = arr[j + 1];
			arr[j + 1] = tmp;
			j--;
		}
	}
	
	for (int i = 0; i < 4; i++) cout << arr[i] << ' ';

	return 0;
}

#include <iostream>

using namespace std;

int main()
{
	cout << "Hello" << endl;
	cout << "Baekjoon" << endl;
	cout << "Online Judge";

	return 0;
}

#include <iostream>

using namespace std;

int main()
{
	char W[10001];
	cin.getline(W, 10001, EOF);
	cout << W;

	return 0;
}
