":w" : 파일 저장
":r" : 파일을 읽어옴
":f" : 파일명 출력
":wq" : 저장 후 빠져나감
"i, o, O, a, A, I" : 편집 위치 지정
## Delete Commands(d+command)

"d" : 라인 삭제
"w" : special character을 삭제(#include 를 #, include로 취급)
"W" : 공백 기준으로 삭제
"}" : 한 문단 지움(d})
"^" : 커서로부터 처음까지 지움
"nx, nX" : 문자 지움(n은 배수)
## Replace Commands

"rx, Rx" : 문자를 x로 치환
"cW" : 단어 단위로 치환
"J" : 라인 축소(줄 삭제)
## Navigation Commands﻿

"h, j, k, l" : 화살표 키 역할
"w, W, e, E" : 문자, 공백 단위 이동
"b, B" : 이전 단어로 이동
"0, $" : 라인의 처음, 끝으로 이동
"^" : 문자의 처음으로 이동
"nG" : n번째 줄로 이동
*": set nu, set number" : 라인 번호 보이게함
## Search Commands

"/or?text" : 앞/뒤 방향으로 검색
검색 후 "n, N" : 검색 단어를 계속 찾음
## Copy & Paste Commands

"[n] + yy, Y" : n 라인 복사
"[n]yW" : n 단어 복사
"p, P" : 붙여넣기
## Substitute/Replace Commands

":[address] s /old/new/[g]
:1,10 -> 1번줄에서 10번 줄
:1,$ -> 파일 전체
:1,. -> 현재까지
:1,$ s/if/IF/ -> 전체에서 if를 IF로 바꿈(한 번)
:1,$ s/if/IF/g -> 전체에서 if를 IF로 바꿈(모두)
* :!cmd : cmd를 임시로 사용해 결과 확인

* exit : 다시 원래 화면으로 전환
