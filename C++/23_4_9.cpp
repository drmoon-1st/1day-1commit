#include <iostream>
#include <cstring>

using namespace std;

int main()
{
  char passwd[11];
  cout << "프로그램을 종료하려면 암호를 입력하세요." << endl;
  while(true){
    cout << "<< 암호 >>";
    cin >> passwd;
    if(strcmp(passwd, "c++") == 0){
      cout << "프로그램을 정상 종료합니다." << endl;
      break;
    }
    else cout << "암호가 틀립니다." << endl;
  }
  return 0;
}



/*

#include <iostream>

using namespace std;

int main()
{
  cout << "이름을 입력하세요: ";
  char name[11];
  cin >> name;

  cout << "이름은 " << name << "입니다." << endl;
  return 0;
}

#####

#include <iostream>

using namespace std;

int main()
{
  char name1[6] = {'G', 'R', 'A', 'C', 'E' , '\n'};
  char name2[5] = {'G', 'R', 'A', 'C', 'E'};

  cout << name1 << ' ' << name2 << endl; 
  return 0;
}

#####

#include <iostream>

using namespace std;

int main()
{
  cout << "너비를 입력하세요: ";
  
  int width;
  cin >> width;

  cout << "높이를 입력하세요: ";

  int height;
  cin >> height;

  int area = width * height;
  cout << "면적은 " << area << endl;
  return 0;
}

#####

#include <iostream>

using namespace std;

int main()
{
  int sum = 0, sum3 = 0;

  for(int i=1; i < 101; ++i){
    sum += i;
    if(i % 3 == 0) sum3 += i;
  }
  cout << sum << ' ' << sum3 << endl;
  return 0;
}
*/