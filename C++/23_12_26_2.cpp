#include <iostream>

int main(){

 long long x = 0;
 int n, a, b;

 scanf("%lld", &x);
 scanf("%d", &n);

 for(int i=0; i < n; i++){
  scanf("%d %d", &a, &b);
  x -= a*b;
 }
 x != 0 ? printf("No\n") : printf("Yes\n");

 return 0;
}