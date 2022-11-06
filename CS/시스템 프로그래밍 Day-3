OS : 시스템 리소스를 관리한다(h/w, s/w 리소스)
멀티 프로그래밍(Multiprogramming) : 하나 이상의 프로세스가 실행 준비중인 상태
프로세스(process) : 실행중인 프로그래밍
프로세스는 ready Q에 들어감 -> OS가 프로세스를 선택 -> CPU에 넣음/실행
ready Q에는 여러 프로세스가 들어갈 수 있음
CPU의 코어가 높을수록 많은 프로세스 처리 가능
io처리 : CPU가 혼자 처리 불가, 프로세스는 다른 입출력 장치에 처리를 요청(system call 함수)
system call : OS에게 프로그래머가 요청할 때 호출하는 함수
Time sharing : 여러 개의 프로세스들이 하나의 CPU 환경에서 동시에 실행되는 것 같은 효과를 내는 것
OS는 실행할 프로세스를 빠르게 switch 해준다(사람 눈에는 여러 프로세스가 동시에 도는 것처럼 보임)
Interrupts
Processor instruction cycle : 프로그램에서 한 instruction이 실행되는 것
이벤트(interrupt) 발생(사용자가 키보드를 누름) -> 시스템 내부에 하드웨어적인 flag가 on인 것이 있는지 OS가 확인 -> OS는 하던 일을 멈추고 interrupt를 걸고 실행 환경 정보를 저장 -> interrupt service routine(발생한 이벤트를 처리하기 위한 루틴)을 불러옴 -> interrupt service routine process가 끝남  -> 멈추었던 부분을 다시 실행
Signal : 이벤트가 발생했다는 소프트웨어적 통지방식
프로세스는 시그널을 받을 수 있음
리눅스에서는 Ctrl + c가 프로세스를 종료함
컴퓨터 시스템은 키보드 이벤트 발생을 OS에 알림 -> OS는 대상 프로그램에 시그널로 알림
threads : 한 프로세스 내에서 여러 개의 task를 동시에 실행하여야 할 경우에는 프로세스 내에 thread를 여러 개 만들어 사용(프로세스의 실행단위)
