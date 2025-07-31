#numpy 기본 기능
#기본 통계 함수를 직접 작성, 평균 분산, 표준편차 구하기
import numpy as np

ss = ['tom', 'james', 'oscat', ]
print(ss, type(ss))    # 리스트
ss2 = np.array(ss)  # 넘파이 배열로 변환
print(ss2, type(ss2))  # 넘파이 배열
#numpy 펑션을 이용하면 배열로 바꿀 수 있다 
#numpy 배열의 특징
#1. N차원 배열을 지원
#2. 다양한 데이터 타입을 지원
#3. 벡터화 연산을 지원하여 빠른 계산 가능
#4. 브로드캐스팅 기능을 통해 서로 다른 크기의 배열 간 연산 가능
#5. 다양한 수학 함수와 통계 함수 제공
#6. 메모리 효율적이며, 대규모 데이터 처리에 적합
#numpy로 기본 통계 함수 구현
ss3 = ['tom', 'james', 'oscat', 5 ]
ss4 = np.array(ss3)  # 넘파이 배열로 변환
print(ss3, type(ss3))    # 리스트
#numpy 배열은 다양한 데이터 타입을 지원하지만, 모든 요소가 동일한 타입이어야 함
#문자열과 숫자가 혼합된 경우, 모든 요소가 문자열로 변환
#우선순위
#1. 문자열
#2. 정수         
#3. 실수
#4. 불리언

#메모리 비교
li = list(range(1, 10))
print(li)
print(id(li[0]), ' ', id(li[1]))  # 리스트 요소의 메모리 주소
arr = np.array(li)
print(li *10) #이런식의 곱하기는 못함 
print('^' * 10)      
print(li * 10, end = ' ')  # 리스트 요소의 메모리 주소
print( )  # 넘파이 배열 요소의 메모리 주소
print([i * 10 for i in li])  # 리스트 요소의 메모리 주소
print('---') 
num_arr = np.array(li)
print(id(num_arr[0]), ' ', id(num_arr[1]))  # 넘파이 배열 요소의 메모리 주소
#주소가 같다 

print(num_arr * 10)  # 넘파이 배열 요소의 곱하기 연산
print('^' * 10)

