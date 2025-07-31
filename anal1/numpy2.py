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


print()
a = np.array([1, 2, 0, 3])
print(a, type(a), a.dtype,  a.shape,   a.ndim, a.size)  # 넘파이 배열의 속성
print(a[0], a[1])  # 넘파이 배열의 요소 접근

b = np.array([[1, 2, 3], [4, 5, 6]])    # 2차원 넘파이 배열
print(b, type(b), b.dtype,  b.shape,   b.ndim, b.size)  # 2차원 넘파이 배열의 속성
print(b.shape, '',  b[0] )  # 2차원 넘파이 차원 떨어뜨리기 

print("\n=== 차원 떨어뜨리기 (인덱싱) ===")
# 첫 번째 행을 선택 -> 2차원에서 1차원으로 떨어짐
print(f"b[0]: {b[0]}")           # [1 2 3]
print(f"b[0].shape: {b[0].shape}")  # (3,) - 1차원 배열
print(f"b[0].ndim: {b[0].ndim}")    # 1차원

print(f"b[1]: {b[1]}")           # [4 5 6]
print(f"b[1].shape: {b[1].shape}")  # (3,) - 1차원 배열

c = np.zeros((2, 2))  # 2x2 크기의 0으로 채워진 배열
print(c)
d = np.ones((2, 2))  # 2x3 크기의 1로 채워진 배열
print(d)
e= np.full((2, 2), 7)  # 2x2 크기의 7로 채워진 배열
print(e)

f = np.eye(2)  # 2x2 단위 행렬
print(f)  # 단위 행렬
print()
print(np.random.rand(50))  # 0과 1 사이의 난수 5개 생성 균등분포 
print(np.random.randn(50))  # 표준 정규 분포에서 난수 5개  음수도 나올 수 있음
print(np.min(np.random.rand(50)))  # 0과 1 사이의 난수 중 최소값
print(np.min(np.random.randn(50)))  # 0과 1 사이의 난수 중 최소값
print(np.random.randint(1, 10, size=(2, 3)))  # 1부터 10 사이의 정수로 이루어진 2x3 배열
#정규분포란  

print(np.random.randn(2,3))  #    # 표준 정규 분포에서 2x3 배열 생성
#딥러닝에서 자주 난수를 발생시킨다. 
np  .random.seed(42)  # 난수 시드 설정 
#시드란 무엇인가?
#시드를 설정하면 난수 생성이 재현 가능해짐
#같은 시드를 사용하면 항상 같은 난수가 생성됨   
print(np.random.rand(2,3))  # 시드가 설정된 상태에서 난수 생성


print("\n=== numpy 배열 인덱싱 ===")

a = np.array([1, 2, 3, 4, 5])  # 1차원 배열 튜플도 가능 
print(f"a: {a}")  # 1차원 배열
print(f"a[0]: {a[0]}")  # 첫 번째 요소
print(f"a[1:4]: {a[1:4]}")  # 슬라이싱 (1부터 3까지)
print(f"a[1:4:2]: {a[1:4:2]}")  #     슬라이싱 (1부터 3까지 2칸씩 건너뛰기)
print(f"a[1:4].shape: {a[1:4].shape}")




