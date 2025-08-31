# 비선형회귀분석 예제 2
# ----------------------------------------------------------------------------------------
# - 선형관계 분석에서는 모델에 다항식이나 교호작용이 포함되면 해석이 직관적이지 않음
# - 결과의 신뢰성이 떨어질 수 있음
# - 선형가정(정규성 등)이 어긋날 때는 다항식 항을 추가한 다항회귀 모델로 대처할 수 있음
#   → 곡선 형태의 회귀선을 사용하여 데이터의 추세를 더 정확하게 반영
# ----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression # 선형, 다항회귀 모델 작성용
from sklearn.preprocessing import PolynomialFeatures # 다항회귀 모델 작성용
np.set_printoptions(suppress=True) # e말고 소수점으로 나옴

# 1. 데이터 준비
x = np.array([257, 270, 294, 320, 342, 368, 396, 446, 480, 580])[:, np.newaxis] # 1차원 -> 2차원 변환
# print("x의 shape: \n", x.shape) # (10, 1)
y = np.array([236, 234, 253, 298, 314, 342, 360, 368, 390, 388])

# 2. 시각화: 곡선의 결과를 띄므로, 선형회귀(직선)로는  정확한 예측이 제한
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()


# ----------------------- 일반 회귀 모델과 다항 회귀 모델 작성 후 비교 -----------------------


# 3. 선형 & 다항회귀 모델 작성
lr = LinearRegression() # 선형회귀(직선) 모델
pr = LinearRegression() # 다항회귀(곡선) 모델

# 4. (선형회귀) 모델 훈련 및 예측
lr.fit(x, y)
y_lin_pred = lr.predict(x)
# print(y_lin_pred)

# 5. (다항회귀) 모델 훈련 및 예측
polyf = PolynomialFeatures(degree=2) # 특징 행렬 생성
x_quad = polyf.fit_transform(x)
pr.fit(x_quad, y)
y_quad_pred = pr.predict(x_quad)
# print(y_quad_pred)

# 6. 선형 & 다항회귀 시각화 비교: 다항회귀가 데이터의 추세를 더 잘 따라가는 모습
plt.scatter(x, y, label='training point')
plt.plot(x, y_lin_pred, label='linear fit', linestyle='--', color='red')
plt.plot(x_quad, y_quad_pred, label='quadratic fit', linestyle='--', color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 7. 예측 성능 점수 비교
print("MSE: 선형: %.3f, 다항: %.3f" % (mean_squared_error(y, y_lin_pred), mean_squared_error(y, y_quad_pred)))
print("설명력: 선형: %.3f, 다항: %.3f" % (r2_score(y, y_lin_pred), r2_score(y, y_quad_pred)))