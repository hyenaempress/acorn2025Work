# 비선형회귀분석 예제 3
# ----------------------------------------------------------------------------------------
# - 보스톤 집값 데이터를 이용해 단순, 다항 회귀 모델 작성
# ----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression # 선형, 다항회귀 모델 작성용
from sklearn.preprocessing import PolynomialFeatures # 다항회귀 모델 작성용

# 1. 데이터 준비
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data", header = None, sep=r"\s+")    
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df)

# 2. 상관계수 확인
print("상관 계수: \n", df.corr()) # [MEDV][LSTAT]: -0.737663(음의 상관관계)

# 3. 필요 데이터 추출
x = df[['LSTAT']].values # 나중에 LinearRegression()에 사용하기 위해 2차원 변환, 하위 계층 비율
y = df['MEDV'].to_numpy() # 주택 가격

# 4. 선형 & 다항회귀(degree 2, degree 3) 모델 작성
model = LinearRegression()

model.fit(x, y)
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)

# 5. 입력용 데이터 생성
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]

# 6. 선형 예측
y_lin_fit = model.predict(x_fit)
# print(y_lin_fit)
model_r2 = r2_score(y, model.predict(x))
print("선형회귀 결정계수(R²) :", model_r2) # 0.54 (54%)

# 7. 다항(degree = 2) 예측
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))
q_r2 = r2_score(y, model.predict(x_quad))
print('다항(2)회귀 결정 계수(R²): ', q_r2) # 0.64 (64%)

# 8. 다항(degree = 3) 예측
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))
c_r2 = r2_score(y, model.predict(x_cubic))
print('다항(3)회귀 결정 계수(R²): ', c_r2) # 0.65 (65%)

# 9. 선형 & 다항회귀 시각화 비교
plt.scatter(x, y, label='학습 데이터', color='lightgray')
plt.plot(x_fit, y_lin_fit, linestyle=":", label="linear fit(d=1), $R²=%.2f$"%model_r2, color='blue', linewidth=2)
plt.plot(x_fit, y_quad_fit, linestyle=":", label="quadratic fit(d=2), $R²=%.2f$"%q_r2, color='red', linewidth=2)
plt.plot(x_fit, y_cubic_fit, linestyle=":", label="cubic fit(d=3), $R²=%.2f$"%c_r2, color='black', linewidth=2)
plt.xlabel('하위 계층 비율')
plt.ylabel('주택 가격')
plt.title('회귀')
plt.legend()
plt.show()
plt.close()