# ML(기계학습 - 지도학습)
# 회귀분석 : 입력 데이터에 대한 잔차 제곱 합이 최소가 되도록
# 이를 통해 독립변수가 종속변수에 얼마나 영향을 주는지 인과관계를 분석
# 독립변수: 연속형
# 종속변수: 연속형
# 두 변수는 상관 관계가 있어야 하며, 인과관계를 보여야 한다.
# 정량적인 모델을 생성,

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
from sklearn.linear_model import LinearRegression
np.random.seed(12)

# 모델 생성 후 맛보기
# 방법1 : make_regression을 사용 : model 안 만들어짐
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x)
print(y)
print(coef) # 89.47430739278907

y_pred = 89.47430739278907 * -1.70073563 + 100
# 회귀식: y = wx + b
print('y_pred: ', y_pred) # 실제값:-52.17214291 예측값: -52.17214255248879

# 미지의 x에 대한 예측값 y 얻기
print('y_pred_new: ', 89.47430739278907 * 5 + 100) # 547.3715369639453

xx = x
yy = y

# 방법2: LinearRegression을 사용: model O
model = LinearRegression()
fit_model = model.fit(xx, yy) # 학습 데이터로 모형 추정. 절편, 기울기를 얻을 수 있다.
print(fit_model.coef_) # 기울기[89.47430739]
print(fit_model.intercept_) # 절편 [100.0]
# y = [89.47430739] * x + [100.0]
print('예측값y[0]:', 89.47430739 * xx[[0]] + 100.0) # -52.17214291
print('예측값y[0]: ', model.predict(xx[[0]])) # -52.17214291
# 미지의 xx(5)에 대한 예측값 y 얻기
print('미지의 x에 대한 예측값 y: ', model.predict([[5]])) # 547.37153696
print('미지의 x에 대한 예측값 y: ', model.predict([[5], [3]])) # 547.37153696 368.42292218

print(xx)

# 방법3: ols 사용 : model O
import statsmodels.formula.api as smf
import pandas as pd
# x1 = xx.flatten()  # 1차원 배열로 변환
x1 = xx.ravel() # 1차원 배열로 변환
print(x1.shape)
y1 = yy

data = np.array([x1, y1])
print(data.T)
df = pd.DataFrame(data.T)
df.columns = ['x1', 'y1']
print(df.head(2))

model2 = smf.ols(formula = 'y1 ~ x1', data=df).fit()

print(model2.summary())
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    100.0000   7.33e-15   1.36e+16      0.000     100.000     100.000
# x1            89.4743   6.48e-15   1.38e+16      0.000      89.474      89.474
# ==============================================================================
# Omnibus:                        7.616   Durbin-Watson:                   1.798
# Prob(Omnibus):                  0.022   Jarque-Bera (JB):                8.746
# Skew:                           0.516   Prob(JB):                       0.0126
# Kurtosis:                       4.770   Cond. No.                         1.26
# ==============================================================================
print(x1[:2]) # [-1.70073563 -0.67794537]
new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]}) # 기존 자료로 DataFrame 생성
print(new_df)
new_pred= model2.predict(new_df)
print(new_pred)

# 전혀 새로운 독립변수로 종속변수 예측
new_df2 = pd.DataFrame({'x1': [123, -2.677]})
new_pred2 = model2.predict(new_df2)
print('new_pred2 :', new_pred2.values) # [11105.33980931  -139.52272089]