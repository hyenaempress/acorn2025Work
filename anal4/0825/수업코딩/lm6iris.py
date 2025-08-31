# 단순 선형 회귀 : ols 사용
# 상관관계가 선형회귀 모델의 미치는 영향에 대해 알아본다.

import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
print(iris.head(2))
# print(iris.corr()) # 데이터셋에 'species' 칼럼이 있어서 오류
print(iris.select_dtypes(include='number').corr()) # 데이터셋에서 숫자형 변수들만 지정해서 상관관계 확인
# print(iris.iloc[:, 0:4].corr())

# 연습 1: 상관관계가 약한(-0.117570) 두 변수(sepal_width, sepal_length)를 사용
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit() # 종속변수, 독립변수 반대로 사용해도 문제 없음(2개 변수 모두 수치형이므로)
print("검정 결과1", result1.summary())
print("결정계수(R^2)", result1.rsquared) # R-squared: 0.0138226(약 0.014) : 설명력이 떨어짐
print("p-value", result1.pvalues.iloc[1]) # Prob(F-statistic): 0.151898 > 0.05 : 유의하지 않은 모델

# 산점도 확인: 데이터가 너무 퍼져있어서 추세선을 그려도 의미가 없음
plt.scatter(iris.sepal_width, iris.sepal_length)
plt.plot(iris.sepal_width, result1.predict(), color='r')
# plt.show()
plt.close()

# 연습 2: 상관관계가 강한(0.871754) 두 변수(sepal_length, petal_length)를 사용
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit() # 종속변수, 독립변수 반대로 사용해도 문제 없음(2개 변수 모두 수치형이므로)
print("검정 결과2", result2.summary())
print("결정계수(R^2)", result2.rsquared) # R-squared: 0.6690276(약 0.669) : 설명력이 떨어짐
print("p-value", result2.pvalues.iloc[1]) # Prob(F-statistic): 1.04e-47 < 0.05 : 유의한 모델

# 산점도 확인: 데이터가 잘 퍼져있어서 추세선이 의미가 있음
plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), color='r')
# plt.show()
plt.close()

# 결정계수로 상관계수 구하기
print("결정계수로 역추론한 상관계수: ", result2.rsquared ** 0.5)

# 실제 값과 예측 값 일부 비교
print('실제 값: ', iris.sepal_length[:5].values)
print('예측 값: ', result2.predict()[:5])

# 새로운 값으로 예측
new_data = pd.DataFrame({'petal_length': [1.1, 0.5, 5.0]})
y_pred = result2.predict(new_data)
print('예측 결과(sepal_length): \n', y_pred)

# 다중 선형회귀: 독립변수가 2개 이상인 경우
print('--다중 선형회귀: 독립변수가 2개 이상인 경우--')
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width + sepal_width', data=iris).fit()
column_select = "+".join(iris.columns.difference(['sepal_length', 'species'])) # species 제외한 모든 칼럼을 독립변수로 사용
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
print("검정 결과3", result3.summary())
print("결정계수(R^2)", result3.rsquared) # 0.856
print("p-value", result3.pvalues.iloc[1]) # 8.59e-62


