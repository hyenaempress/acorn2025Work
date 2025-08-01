# numpy array() 관련 연습문제
import numpy as np

# 1) step1 : array 관련 문제
print("--- step1: 정규분포 난수 5x4 배열, 행별 합계/최댓값 ---")
arr1 = np.random.randn(5, 4)
for i, row in enumerate(arr1):
    print(f"{i+1}행 합계   : {row.sum()}")
    print(f"{i+1}행 최댓값 : {row.max()}")

# 2) step2 : indexing 관련문제
print("\n--- step2-1: 6x6 zero 행렬, indexing ---")
arr2 = np.zeros((6, 6))
arr2.flat[:] = np.arange(1, 37)
print("2번째 행 전체:", arr2[1])
print("5번째 열 전체:", arr2[:, 4])
print("15~29 추출:")
print(arr2[2:5, 2:5])

print("\n--- step2-2: 6x4 zero 행렬, 난수로 초기화 및 수정 ---")
arr3 = np.zeros((6, 4))
rand_start = np.random.randint(20, 101, 6)
for i in range(6):
    arr3[i] = np.arange(rand_start[i], rand_start[i]+4)
print(arr3)
arr3[0] = 1000
arr3[-1] = 6000
print(arr3)

# 3) step3 : unifunc 관련문제
print("\n--- step3: 4x5 정규분포 배열, 기술통계량 ---")
arr4 = np.random.randn(4, 5)
print(arr4)
print("평균:", np.mean(arr4))
print("합계:", np.sum(arr4))
print("표준편차:", np.std(arr4))
print("분산:", np.var(arr4))
print("최댓값:", np.max(arr4))
print("최솟값:", np.min(arr4))
print("1사분위수:", np.percentile(arr4, 25))
print("2사분위수:", np.percentile(arr4, 50))
print("3사분위수:", np.percentile(arr4, 75))
print("요소값 누적합:", np.cumsum(arr4))

# Q1) 브로드캐스팅과 조건 연산
print("\n--- Q1: 브로드캐스팅과 조건 연산 ---")
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
mul = a * b
print(mul)
print("30 이상:", mul[mul >= 30])

# Q2) 다차원 배열 슬라이싱 및 재배열
print("\n--- Q2: 슬라이싱 및 재배열 ---")
arr5 = np.arange(1, 13).reshape(3, 4)
print("2번째 행:", arr5[1])
print("1번째 열:", arr5[:, 0])
arr5_r = arr5.reshape(4, 3)
print("(4,3)로 reshape:", arr5_r)
print("flatten:", arr5_r.flatten())

# Q3) 1~100 배열에서 3의 배수이면서 5의 배수 아닌 값 추출, 제곱
print("\n--- Q3: 조건 추출 및 제곱 ---")
arr6 = np.arange(1, 101)
cond = (arr6 % 3 == 0) & (arr6 % 5 != 0)
filtered = arr6[cond]
squared = filtered ** 2
print(squared)

# Q4) 조건에 따라 문자열/값 변환
print("\n--- Q4: 조건에 따라 문자열/값 변환 ---")
arr7 = np.array([15, 22, 8, 19, 31, 4])
str_arr = np.where(arr7 >= 10, 'High', 'Low')
print(str_arr)
arr7_mod = arr7.copy()
arr7_mod[arr7_mod >= 20] = -1
print(arr7_mod)

# Q5) 정규분포 난수 1000개, 상위 5% 값 출력
print("\n--- Q5: 정규분포 난수, 상위 5% ---")
data = np.random.normal(50, 10, 1000)
thresh = np.percentile(data, 95)
print(data[data >= thresh])
