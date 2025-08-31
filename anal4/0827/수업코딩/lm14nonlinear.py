# 비선형회귀분석 예제 1
# ----------------------------------------------------------------------------------------
# - 선형관계 분석에서는 모델에 다항식이나 교호작용이 포함되면 해석이 직관적이지 않음
# - 결과의 신뢰성이 떨어질 수 있음
# - 선형가정(정규성 등)이 어긋날 때는 다항식 항을 추가한 다항회귀 모델로 대처할 수 있음
#   → 곡선 형태의 회귀선을 사용하여 데이터의 추세를 더 정확하게 반영
# ----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # 다항회귀 모델 작성용

# 1. 데이터 준비
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3 ,7])

# 2. 시각화: 곡선의 결과를 띄므로, 선형회귀(직선)로는 정확한 예측이 제한
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

# 3. 상관 계수
print("상관 계수: \n", np.corrcoef(x, y)) # 0.48

# 4. 선형회귀 모델 작성 전 model = LinearRegression()의 입력 형태에 맞추어 1차원 -> 2차원 데이터 변환
x = x[:, np.newaxis]

# 5. 선형 회귀 모델 작성 및 예측
model = LinearRegression().fit(x, y)
y_pred = model.predict(x)
print("선형회귀 예측값 : \n", y_pred) # 2.0 2.7 3.4 4.1 4.8
print("선형회귀 결정계수(R²) : \n", r2_score(y, y_pred)) # 0.23 (23%: 나쁘지 않음)

# 6. 선형회귀(직선) 예측 결과 시각화: 예측이 직선 형태로, 실제 데이터는 곡선 형태로 분포되어 있어 오차가 큼
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# -----------------------따라서, 곡선 형태로 예측하는 다항회귀 모델이 필요하다-----------------------


# 7. 다항회귀 모델 작성 - 추세선의 유연성을 위해 열 추가
poly = PolynomialFeatures(degree=2, include_bias=False) # degree: 열의 수, include_bias: 절편 포함 여부

# - 7.1 특징 행렬 생성
x2 = poly.fit_transform(x) # x[1, 2, 3, 4, 5]를 ^2 한 행렬이 생성 됌
# print(x2)

# - 7.2 모델 작성
model2 = LinearRegression().fit(x2, y)
ypred2 = model2.predict(x2)
print("다항회귀 예측값 : \n", ypred2.round(1)) # 4.1 1.6 1.2 3.0 6.9
print("다항회귀 결정계수(R²) : \n", r2_score(y, ypred2)) # 0.98 (98%: 매우 좋음, 과적합(overfitting) 우려)

# 8. 다항회귀(곡선) 예측값 시각화: 예측이 곡선 형태로 실제 데이터의 분포와 추세를 잘 따름
# - 7번의 degree(2 -> 4 -> 6) 값을 올릴 수록, 더욱 정밀하게 예측함
plt.scatter(x, y)
plt.plot(x, ypred2, color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 결론: 데이터의 분포가 곡선을 띄고 있다면, 다항회귀 모델을 사용하는 것이 더 적합하다.