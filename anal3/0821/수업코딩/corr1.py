# 공분산, 상관계수
# 공분산: 두 변수의 패턴을 확인하기 위해 공분산을 사용. 단위 크기에 영향을 
# 상관계수: 공분산을 표준화. -1 ~ 0 ~ 1. +-1에 근사하면 관계가 강함
import numpy as np
import matplotlib.pyplot as plt

# 두 배열의 공분산을 계산 (1~5, 2~6)
print(np.cov(np.arange(1, 6), np.arange(2, 7)))
# 두 배열의 공분산을 계산 (10~50, 20~60)
print(np.cov(np.arange(10, 60, 10), np.arange(20, 70, 10)))
# 두 배열의 공분산을 계산 (100~500, 200~600)
print(np.cov(np.arange(100, 600, 100), np.arange(200, 700, 100)))
# 한 배열이 모두 같은 값일 때 공분산 계산
print(np.cov(np.arange(1, 6), (3, 3, 3, 3, 3)))
# 두 배열이 반대 방향일 때 공분산 계산 (1~5, 6~2)
print(np.cov(np.arange(1, 6), np.arange(6, 1, -1)))
print("-------------")

x = [8, 3, 6, 6, 9, 4, 3, 9, 3, 4]  # x 데이터
print('x의 평균:', np.mean(x))  # x의 평균 출력
print('x의 분산:', np.var(x))  # x의 분산 출력

y = [60, 20, 40, 60, 90, 50, 10, 80, 40, 50]  # y 데이터
print('y의 평균:', np.mean(y))  # y의 평균 출력
print('y의 분산:', np.var(y))  # y의 분산 출력

# x, y의 산점도 그리기 (주석 처리됨)
# plt.scatter(x, y)
# plt.show()

print('x, y 공분산: ', np.cov(x, y))  # x, y의 공분산 행렬 출력
print('x, y 공분산: ', np.cov(x, y)[0, 1])  # x, y의 공분산 값만 출력
print()
print('x, y 상관계수: ', np.corrcoef(x, y))  # x, y의 상관계수 행렬 출력
print('x, y 상관계수: ', np.corrcoef(x, y)[0, 1])  # x, y의 상관계수 값만 출력

# 참고: 비선형인 경우는 일반적인 상관계수 방법을 사용하면 안됨
m = [-3, -2, -1, 0, 1, 2, 3]  # m 데이터 (비선형)
n = [9, 4, 1, 0, 1, 4, 9]     # n 데이터 (비선형)
plt.scatter(m, n)              # m, n의 산점도 그리기
plt.show()                     # 그래프 표시
print('m, n 상관계수 : ', np.corrcoef(m, n)[0, 1]) # 비선형 데이터의 상관계수 출력 (의미 없음)

