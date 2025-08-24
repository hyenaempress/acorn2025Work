# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용

# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
#  - 1)지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 운동 시간 / 요구사항: 회귀분석 모델 작성
#  - 2)지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 종편 시청 시간 / 요구사항: 회귀분석 모델 작성
#     참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.
#      - 결측치: 해당 칼럼의 평균 값을 사용 / 이상치: 행 제거 / 이상치 기준: 10시간 초과  

# 구분,지상파,종편,운동
# 1,0.9,0.7,4.2
# 2,1.2,1.0,3.8
# 3,1.2,1.3,3.5
# 4,1.9,2.0,4.0
# 5,3.3,3.9,2.5
# 6,4.1,3.9,2.0
# 7,5.8,4.1,1.3
# 8,2.8,2.1,2.4
# 9,3.8,3.1,1.3
# 10,4.8,3.1,35.0
# 11,NaN,3.5,4.0
# 12,0.9,0.7,4.2
# 13,3.0,2.0,1.8
# 14,2.2,1.5,3.5
# 15,2.0,2.0,3.5

from xml.parsers.expat import model
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # ols
from sklearn.linear_model import LinearRegression

# Data 구성
data = pd.DataFrame({
    '구분': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    '지상파': [0.9,1.2,1.2,1.9,3.3,4.1,5.8,2.8,3.8,4.8,np.nan,0.9,3.0,2.2,2.0],
    '종편': [0.7,1.0,1.3,2.0,3.9,3.9,4.1,2.1,3.1,3.1,3.5,0.7,2.0,1.5,2.0],
    '운동': [4.2,3.8,3.5,4.0,2.5,2.0,1.3,2.4,1.3,35.0,4.0,4.2,1.8,3.5,3.5]
})

# 결측치 처리(해당 칼럼의 평균 값으로 대체)
data['지상파'].fillna(data['지상파'].mean(), inplace=True)

# 이상치 처리(행 제거)
data = data[data['운동'] <= 10] # 10시간 미만만 남김

# 문제 1) scipy.stats.linregress()
#  - 1.1)지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 운동 시간 / 요구사항: 회귀분석 모델 작성
print("\n1.1------------------------------------------------------------------------")
# 1. 모델 작성
print("<1.1.1 모델 작성 정보>")
model1 = stats.linregress(data['지상파'], data['운동'])
print("기울기(slope):", model1.slope)
print("절편(intercept):", model1.intercept)
print("상관계수(rvalue):", model1.rvalue)
print("결정계수(R²):", model1.rvalue**2)
print("p-value:", model1.pvalue)
print("표준오차(stderr):", model1.stderr)
print("절편 표준오차(intercept_stderr):", model1.intercept_stderr)

print("\n<1.1.2 모델 예측 결과>")
# 2. 점수 예측
print('(지상파 2시간) 운동 시간 예측: ', model1.slope * 2 + model1.intercept) # 3.3727659863595 vs (실제) 3.5
print('(지상파 선두 5개) 운동 시간 예측: ', np.polyval([model1.slope, model1.intercept], np.array(data['지상파'][:5])))
print('실제 운동 시간: ', data['운동'][:5].values)

#  - 1.2)지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 종편 시청 시간 / 요구사항: 회귀분석 모델 작성
print("\n1.2------------------------------------------------------------------------")
# 1. 모델 작성
print("<1.2.1 모델 작성 정보>")
model2 = stats.linregress(data['지상파'], data['종편'])
print("기울기(slope):", model2.slope)
print("절편(intercept):", model2.intercept)
print("상관계수(rvalue):", model2.rvalue)
print("결정계수(R²):", model2.rvalue**2)
print("p-value:", model2.pvalue)
print("표준오차(stderr):", model2.stderr)
print("절편 표준오차(intercept_stderr):\n", model2.intercept_stderr)

print("\n<1.2.2 모델 예측 결과>")
# 2. 점수 예측
print('(지상파 2시간) 종편 시청 시간 예측: ', model2.slope * 2 + model2.intercept)
print('(지상파 선두 5개) 종편 시청 시간 예측: ', np.polyval([model2.slope, model2.intercept], np.array(data['지상파'][:5])))
print('실제 종편 시청 시간: ', data['종편'][:5].values)

# 문제 2) statsmodels ols()
#  - 2.1)지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 운동 시간 / 요구사항: 회귀분석 모델 작성
print("\n2.1------------------------------------------------------------------------")
# 1. 모델 작성
model3 = smf.ols(formula = '운동 ~ 지상파', data = data).fit()
print("<2.1.1 모델 작성 정보>")
print("기울기(slope):", model3.params[1])
print("절편(intercept):", model3.params[0])
print("결정계수(R²):", model3.rsquared)
print("p-value:", model3.pvalues[1])
print("표준오차(stderr):", model3.bse[1])
print("절편 표준오차(intercept_stderr):", model3.bse[0])

print("\n<2.1.2 모델 예측 결과>")
# 2. 점수 예측
print('(지상파 2시간) 운동 시간 예측: ', model3.predict(pd.DataFrame({'지상파': [2]}))[0])
print('(지상파 선두 5개) 운동 시간 예측: \n', model3.predict(data[['지상파']][:5]))
print('실제 운동 시간: ', data['운동'][:5].values)

#  - 2.2)지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 종편 시청 시간 / 요구사항: 회귀분석 모델 작성
print("\n2.2------------------------------------------------------------------------")
# 1. 모델 작성
model4 = smf.ols(formula = '종편 ~ 지상파', data = data).fit()
print("<2.2.1 모델 작성 정보>")
print("기울기(slope):", model4.params.iloc[1])
print("절편(intercept):", model4.params.iloc[0])
print("결정계수(R²):", model4.rsquared)
print("p-value:", model4.pvalues.iloc[1])
print("표준오차(stderr):", model4.bse.iloc[1])
print("절편 표준오차(intercept_stderr):", model4.bse.iloc[0])

print("\n<2.2.2 모델 예측 결과>")
# 2. 점수 예측
print('(지상파 2시간) 종편 시간 예측: ', model4.predict(pd.DataFrame({'지상파': [2]}))[0])
print('(지상파 선두 5개) 종편 시간 예측: \n', model4.predict(data[['지상파']][:5]))
print('실제 종편 시간: ', data['종편'][:5].values)

# 문제 3) LinearRegression
#  - 3.1)지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 운동 시간 / 요구사항: 회귀분석 모델 작성
print("\n3.1------------------------------------------------------------------------")
model5 = LinearRegression().fit(data[['지상파']], data['운동'])
print("<3.1.1 모델 작성 정보>")
# # 1. 모델 작성
print("기울기(slope):", model5.coef_[0])
print("절편(intercept):", model5.intercept_)
print("결정계수(R²):", model5.score(data[['지상파']], data['운동']))

print("\n<3.1.2 모델 예측 결과>")
# 2. 점수 예측
print('(지상파 2시간) 운동 시간 예측: ', model5.predict(pd.DataFrame({'지상파': [2]}))[0])
print('(지상파 선두 5개) 운동 시간 예측: ', model5.predict(data[['지상파']][:5]))
print('실제 운동 시간: ', data['운동'][:5].values)

#  - 3.2)지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#      - 입력: 지상파 시청 시간 / 출력: 종편 시청 시간 / 요구사항: 회귀분석 모델 작성
print("\n3.2------------------------------------------------------------------------")
model6 = LinearRegression().fit(data[['지상파']], data['종편'])
# # 1. 모델 작성
print("<3.2.1 모델 작성 정보>")
print("기울기(slope):", model6.coef_[0])
print("절편(intercept):", model6.intercept_)
print("결정계수(R²):", model6.score(data[['지상파']], data['종편']))

print("\n<3.2.2 모델 예측 결과>")
# 2. 점수 예측
print('(지상파 2시간) 종편 시간 예측: ', model6.predict(pd.DataFrame({'지상파': [2]}))[0])
print('(지상파 선두 5개) 종편 시간 예측: ', model6.predict(data[['지상파']][:5]))
print('실제 종편 시간: ', data['종편'][:5].values)
