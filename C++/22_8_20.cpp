#include <iostream>

using namespace std;

int main()
{
	int tmp;
	int arr[10] = {1, 2, 4, 3, 6 ,5, 7, 8, 9, 10};

	//ios::sync_with_stdio(0);
	//cin.tie(0);

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 9 - i; j++) {
			if (arr[j] > arr[j+1]) {
				tmp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = tmp;
			}
		}
	}

	for (int i = 0; i < 10; i++) {
		printf("%d", arr[i]);
	}

	return 0;
}

#include <iostream>

using namespace std;

int arr[500000];

int main()
{
	int tmp, N, sum=0;
	

	ios::sync_with_stdio(0);
	cin.tie(0);

	cin >> N;

	for (int i = 0; i < N; i++) cin >> arr[i];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N - 1 - i; j++) {
			if (arr[j] > arr[j + 1]) {
				tmp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = tmp;
			}
		}
	}

	int count = NULL, cnt1 = 1, idx1 = 0, cnt2 = 1, idx2 = 0;

	for (int i = 0; i < N; i++) {
		int cnt = 0, idx = i;
		sum += arr[i];
		if (count = arr[i]) continue;
		else count = arr[i];

		for (int j = i + 1; j < N; j++) {
			if (count != arr[j]) {
				if (cnt1 >= cnt && cnt2 <= cnt) {
					cnt2 = cnt;
					idx2 = idx;
					break;
				}
				cnt += 1;
			}
			if (cnt > cnt1) cnt1 = cnt;
		}
	}
	cout << "########" << endl;
	cout << sum / N << endl;
	cout << arr[N / 2] << endl;
	cout << arr[idx2] << endl;
	cout << arr[N - 1] - arr[0];

	return 0;
}
