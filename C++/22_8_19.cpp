#include <iostream>

using namespace std;

int main()
{
	int i, j, min, idx, tmp;
	int array[10] = { 1, 2, 3, 4, 6, 5, 7, 8, 9, 10 };
	for (i = 0; i < 10; i++) {
		min = 9999;
		for (j = i; j < 10; j++) {
			if (min < array[j]) {
				min = array[j];
				idx = j;
			}
		}
		tmp = array[i];
		array[i] = array[idx];
		array[idx] = tmp;
	}

	return 0;
}

#include <iostream>

using namespace std;

int main()
{
	int a, b;
	int arr[1000];

	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> a >> b;
	for (int i = 0; i < a; i++) cin >> arr[i];
	for (int i = 0; i < a; i++) {
		for (int j = i+1; j < a; j++) {
			if (arr[i] > arr[j]) {
				int tmp;
				tmp = arr[j];
				arr[j] = arr[i];
				arr[i] = tmp;
			}
		}
	}
	cout << arr[a - b];
	return 0;
}