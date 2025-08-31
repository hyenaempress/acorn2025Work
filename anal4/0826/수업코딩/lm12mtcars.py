# Linear Regression으로 선형회귀 모델 작성 - mtcars

import statsmodels.api
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


mtcars = statsmodels.api.datasets.get_rdataset("mtcars").data
print(mtcars)
print(mtcars.corr(method='pearson'))
print()

x = mtcars[['hp']].values
print(x[:3])
y = mtcars[['mpg']].values
print(y[:3])

lmodel = LinearRegression().fit(x, y)
print("slope : ", lmodel.coef_) # -0.06822828
print("intercept : ", lmodel.intercept_) # 30.09886054

# 산포도
# plt.scatter(x, y)
# plt.plot(x, lmodel.coef_ * x + lmodel.intercept_, c='r')
# plt.xlabel('Horsepower')
# plt.ylabel('Miles per Gallon')
# plt.title('Linear Regression: MPG vs Horsepower')
# plt.show()

pred = lmodel.predict(x)
print("예측 값: \n", pred[:5].round(3))
print("실제 값: \n", y[:5].round(3))

# 모델 성능 평가
# - **MSE (Mean Squared Error, 평균 제곱 오차)**
print('MSE(평균제곱오차) : {}'.format(mean_squared_error(y, pred))) # 13.989822298268805
print('R^2_score(결정계수) : {}'.format(r2_score(y, pred))) # 0.602437341423934

# 새로운 마력 수에 대한 연비 예측
new_hp = [[123]]
new_pred = lmodel.predict(new_hp)
print("%s 마력인 경우 연비는 약 %s 입니다" % (new_hp[0][0], new_pred[0][0]))