#배열연산 
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x , x.astype, x.dtype)  # 배열 출력, 타입 확인
#배열 연산은 무엇일까?
#배열 연산은 배열의 요소들에 대해 수학적 연산을 수행하는 것
#예를 들어, 배열의 모든 요소에 2를 더하는 연산은 배열의 각 요소에 2를 더하는 것을 의미
#배열 연산은 벡터화된 연산으로, 반복문 없이 빠르게 수행 가능
#배열 연산은 NumPy의 핵심 기능 중 하나로, 데이터 분석과 머신러닝에서 매우 중요
#배열 연산의 장점은 무엇일까?       
#1. **속도**: 반복문 없이 빠르게 수행 가능                  
#2. **간결성**: 코드가 간단하고 읽기 쉬움
#3. **메모리 효율성**: 불필요한 복사 없이 데이터 접근
#4. **라이브러리 호환성**: Pandas, Sklearn 등 모든 라이브러리의 기초            
#5. **배열 연산의 기초**: 배열의 요소들에 대해 수학적 연산을 수행하는 것
#배열 연산의 기초는 무엇일까?       
#배열 연산의 기초는 배열의 요소들에 대해 수학적 연산을 수행하는 것
#예를 들어, 배열의 모든 요소에 2를 더하는 연산은 배열           
#의 각 요소에 2를 더하는 것을 의미
#배열 연산은 벡터화된 연산으로, 반복문 없이 빠르게 수행 가능    
#배열 연산의 기초는 NumPy의 핵심 기능 중 하나로, 데이터 분석과 머신러닝에서 매우 중요           
#배열 연산의 기초는 무엇일까?
#배열 연산의 기초는 배열의 요소들에 대해 수학적 연산을 수행하는 것
#예를 들어, 배열의 모든 요소에 2를 더하는 연산은 배열      


x = np.array([[1, 2], [3, 4]], dtype=np.float64)  # 배열 생성, 타입 지정
# x= np.array(5,9).reshape(2, 2)  # 3x3 배열 생성 
y = np.array([5, 9, 7, 1]).reshape(2, 2)  # 리스트로 감싸기 + 4개 요소 필요


# 여기 문법 오류 생겼는데 나중에 고칠게 
#리쉐입을 사용해서 차원을 변경하였음 
y = y.astype(np.float64)  # 타입 변경

#배열 더하기 
print("배열 더하기:")
print(x + y)  # 배열의 각 요소에 5를 더함
print(np.add(x, y))  # NumPy의 add 함수 사용

#유니버셜 함수?
#유니버셜 함수는 NumPy에서 제공하는 함수로, 배열의 각 요소에 대해 연산을 수행하는 함수

#np.subtract, mp.multiply, np.divide 
#더하기 빼기 곱하기 나누기 add랑 차이가 있다. 
import time
big_arr= np.random.rand(1000000) #            100만 개의 랜덤 숫자 생성
start_time = time.time()  # 시작 시간 기록
sum(big_arr)  #파이썬 함수 
end = time.time()  # 종료 시간 기록
print(f"sum() : {end - start_time:.6f}sec")  # 소요 시간 출력


#요소별 곱하기 
print("배열 곱하기:")
print(x * y)  # 배열의 각 요소를 곱함
#머신러닝을 할려면 곱하기 필수 
print(np.multiply(x, y))  # NumPy의 multiply 함수 사용
print(x.dot(y))  # 행렬 곱셈 내적연산 
print()
#내적연산이란?
#내적연산은 두 벡터의 곱을 계산하는 연산으로, 벡터의 방향과 크기를 고려하여 결과를 도출
#내적연산은 머신러닝에서 자주 사용되는 연산으로, 벡터의 유사도를 계산하는 데 사용
#내적연산의 장점은 무엇일까?
#1. **유사도 계산**: 벡터의 유사도를 계산하여 데이터 간의 관계를 파악
#2. **차원 축소**: 고차원 데이터를 저차원으로 변                
#3. **효율성**: 벡터의 방향과 크기를 고려하여 빠르게 계산
#4. **머신러닝 모델**: 많은 머신러닝 모델에서 핵심          
#5. **선형대수학**: 선형대수학의 기초로, 벡터와 행렬의 연산을 이해하는 데 중요
#6. **다양한 응용**: 이미지 처리, 자연어 처리 등 다양한 분야에서 활용
#아다마르곱, 내적

#8월 1일 교육 

v = np.array([9,10])# 아다마르 곱 (요소별 곱셈)
w = np.array([11,12])
print("아다마르 곱:")
print(v * w)  # 요소별 곱셈
print(v.dot(w))  # 내적 연산
print(np.multiply(v, w))  # NumPy의 multiply 함수 사용 이쪽이 훨씬 빠르게 값이 나옴 C를 운영중이기 떄문에
#성분 곱이 가능하다.
#양수가 나왔으니 그래프상에서 둔각은 아니다 
#아다마르 곱은 무엇일까?
#아다마르 곱은 두 벡터의 각 요소를 곱하는       
#연산으로, 벡터의 방향과 크기를 고려하지 않고 요소별로 곱함
#아다마르 곱은 머신러닝에서 자주 사용되는 연산으로, 벡터의 유사도를 계산하는 데 사용
#아다마르 곱의 장점은 무엇일까?
#1. **요소별 연산**: 벡터의 각 요소를 독립적으로 곱하여 계산
#2. **간단한 계산**: 벡터의 방향과 크기를 고려          
#3. **효율성**: 빠르게 계산 가능
#4. **머신러닝 모델**: 많은 머신러닝 모델에서 핵심      
#5. **선형대수학**: 선형대수학의 기초로, 벡터와 행렬의 연산을 이해하는 데 중요
#6. **다양한 응용**: 이미지 처리, 자연어 처리 등 다양한 분야에서 활용


print(np.dot(v, w))  # 내적 연산
print(np.dot(x, v))  # x와 v의 내적 연산
print(np.dot(x, y))  # x와 y의 내적 연산 2행2열임으로 스칼라값이 나옴

print   ("=== 유용한 함수 ===")
print(x)
print(np.sum(x))  # 배열의 모든 요소의 합
print(np.sum(x, axis= 0)) # 각 열의 합
print(np.sum(x, axis=1))  # 각 행의 합
#축이 없으면 전체 더하기를 해야하지만 축을 설정해준것 
print(np.min(x), '' , np.max (x))  # 배열의 평균
print(np.argmax(x), '' , np.atgmax (x))  # 배열의 최대값과 최소값의 인덱스
#인덱스를 리턴하는데 그 인덱스가 많이 사용됨 
print(np.cumsum(x))  # 누저적 합
print(np.cumprod(x))  # 누적 곱
print() 
print("=== 문자열 배열 ===")
names = np.array(['tom', 'james', 'oscar', 'tom', 'oscar'])
names2 = np.array(['tom', 'page', 'jhon'])

print(np.unique(names))  # 중복 제거
print(np.intersect1d(names, names2))  # 교집합
print(np.union1d(names, names2))  # 합집합  
print(np.setdiff1d(names, names2))  # 차집합
print(np.setdiff1d(names, names2, assume_unique=True))  # 차집합
print(np.in1d(names, names2))  # 포함 여부
print()  

print("=== 전치 트랜스포즈 행과 열을 바꾸어줌 ===")
print(x)
print(x.T)  # 전치 행렬
print(np.transpose(x))  # 전치 행렬
print(np.linalg.inv(x))  # 역행렬 (x가 정방행렬일 때만 가능)
print(np.linalg.det(x))  # 행렬식 (determinant)
print(np.linalg.eig(x))  # 고유값과 고유벡터

arr = np.arange(1,15).reshape(3, 5)  # 1부터 14까지의 숫자를 3x5 배열로 생성
print(arr)              
print("원본 배열:")
print(arr.T)    # 전치 행렬
print(np.dot(arr, arr.T))  # 그냥 arr만 있으면 안되는데, T를 넣으면서 내적이 가능해진다


#차원 축소 2차원을 1차원으로 떨어뜨림 
print(arr.flatten())  # 1차원 배열로 변환
print(arr.ravel())  # 1차원 배열로 변환 (메모리를 공유)
print(arr.reshape(1, -1))  # 1행으로 변환
print(arr.reshape(-1, 1))  # 1열로 변환








