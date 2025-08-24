# 최소 제곱해를 선형 행렬 방정식으로 구하기 
import numpy as np                   # numpy 라이브러리 불러오기
import matplotlib.pyplot as plt      # matplotlib 라이브러리 불러오기
plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정
import numpy.linalg as lin           # numpy의 선형대수 모듈 불러오기

x = np.array([0, 1, 2, 3])           # x 데이터 배열 생성
y = np.array([-1, 0.2, 0.5, 2.1])    # y 데이터 배열 생성

# plt.scatter(x, y)
# plt.show()

A = np.vstack([x, np.ones(len(x))]).T # x와 1로 구성된 행렬 생성 (회귀식용)
print(A)                              # 행렬 A 출력

# y = wx + b 라는 일차 방정식의 w, b?
w, b = lin.lstsq(A, y, rcond=None)[0] # 최소 제곱법으로 w(기울기), b(절편) 계산
# 최소 제곱법: 잔차 제곱의 총합이 최소가 되는 값을 얻을 수 있다.
print('w(Weight, 기울기, slope):', w) # 기울기 출력
print('b(bias, 절편, 편향, intercept):', b) # 절편 출력

# y = 0.95999 * x + -0.98999 # 단순 선형 회귀 수식(모델)
plt.scatter(x, y)                     # 실제 데이터 산점도 그리기
plt.plot(x, w * x + b, label = '실제값') # 회귀선 그리기
plt.legend()                          # 범례 표시
plt.show()                            # 그래프 표시

# 수식을 사용하여 예측값 구하기
print('예측값:', w * 1 + b)           # x=1일 때 예측값 출력
# -0.02999(예측 값) - 0.2(실제 값): 잔차, 오차, 손실, 에러
# 100% 확정은 아니지만, 어느 정도는 믿을 수 있음