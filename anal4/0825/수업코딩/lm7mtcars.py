# 선형회귀분석 - ols() 사용
# mtcars dataset을 사용 - 독립 변수가 종속 변수(mpg, 연비)에 미치는 영향 분석

import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = statsmodels.api.datasets.get_rdataset("mtcars").data
# print(mtcars)
print(mtcars.columns) # ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(mtcars.info())
print(mtcars.corr())

# 상관 계수 확인
print(np.corrcoef(mtcars.hp, mtcars.mpg)[0, 1]) # -0.7761683
print(np.corrcoef(mtcars.wt, mtcars.mpg)[0, 1]) # -0.8676593

# 산점도 확인: 데이터가 너무 퍼져있어서 추세선을 그려도 의미가 없음
plt.scatter(mtcars.hp, mtcars.mpg)
plt.xlabel('hp')
plt.ylabel('mpg')
# plt.show()
plt.close()

# 단순 선형 회귀: hp -> mpg
print("\n 단순 선형 회귀: hp -> mpg")
result1 = smf.ols(formula='mpg ~ hp', data=mtcars).fit()
print(result1.summary())

print("마력수 110에 대한 연비 예측: ", -0.0682 * 110 + 30.0989) # 22.5969
print("마력수 50에 대한 연비 예측: ", -0.0682 * 50 + 30.0989) # 26.6889

print("마력수 110에 대한 연비 예측: ", result1.predict(pd.DataFrame({'hp': [110]})))
print("마력수 50에 대한 연비 예측: ", result1.predict(pd.DataFrame({'hp': [50]})))

# 다중 선형 회귀: hp, wt -> mpg
print("\n 다중 선형 회귀: hp, wt -> mpg")
result2 = smf.ols(formula='mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary())

