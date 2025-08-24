# ols가 제공하는 표에 대해 알아보자
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')
print(df)

print("회귀분석 실행-----------------")

model = smf.ols(formula='만족도 ~ 적절성', data=df).fit()

print(model.summary())

print('parameters: ', model.params)
print('R squared: ', model.rsquared)
print('p-value: ', model.pvalues)
print('predicted value: ', model.predict())
print('실제값 : \n', df['만족도'], '\n예측값 : \n', model.predict())

plt.scatter(df['적절성'], df['만족도'], label='실제값')
slope, intercept= np.polyfit(df.적절성, df.만족도, 1)