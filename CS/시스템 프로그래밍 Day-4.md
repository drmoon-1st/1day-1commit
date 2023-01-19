program : 어떤 task를 수행하기 위해서 정해진 instruction의 나열
thread : 프로세스 내에서 하나의 시행 흐름을 나타내는 데이터 타입
porgram counter : 다음에 수행할 instruction을 가르키는 것, thread마다 가지고 있음
thread 간의 switch는 process간의 switch보다 간단함(서로 일부분을 공유하기 때문)
"env" : 리눅스에서 환경 변수를 볼 수 있음
Activation record : 함수가 실행되는 동안 함수가 실행하기 위해 필요한 정보가 담겨있는 공간
SYNOPSIS box : 함수의 이름, 파라미터, return 값을 담아 놓는 것
Error : return값이 -1이나 null일 때 대부분 error이 발생했다고 볼 수 있음
void perror(const char s*) : 사용자가 준 string 뒤에 에러 메세지 출력
char* strerror(int errnum) : 에러 코드를 출력
strtok() : string의 토큰으로 분리
