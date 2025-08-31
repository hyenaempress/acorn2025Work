# 선형회귀분석 - ols 사용 : 회귀분석 선행

# *** 선형회귀분석의 기존 가정 충족 조건 ***
# . 선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.
# . 정규성 : 잔차항(오차항)이 정규분포를 따라야 한다.
# . 독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.(여성 데이터를 이용하여 유전 정보를 조사 -> 여성 데이터의 여성이 서로 가족이면 안 됌)
# . 등분산성 : 그룹간의 분산이 유사해야 한다. 독립변수의 모든 값에 대한 오차들의 분산은 일정해야 한다.
# . 다중공선성 : 다중회귀 분석 시 두 개 이상의 독립변수 간에 강한 상관관계가 있어서는 안된다.(독립 변수끼리)

# Advertising.csv : 각 매체의 광고비에 따른 판매량 관련
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import shapiro
import statsmodels.api as sm # Quantiel-Quantile plot 지원
import seaborn as sns
from statsmodels.stats.diagnostic import linear_reset # 모형 적합성
from statsmodels.stats.outliers_influence import OLSInfluence

advdf = pd.read_csv('./01_big_data_machine_learning/data/Advertising.csv', usecols=[1, 2, 3, 4])
print(advdf)
print(advdf.shape)
print(advdf.index, ' ', advdf.columns)
print(advdf.info())

print(advdf.corr())
print()

# 단순 선형 회귀 모델: x:tv, y:sales
lmodel1 = smf.ols(formula='sales ~ tv', data=advdf).fit()
print(lmodel1.params)
print(lmodel1.rsquared)
# print(lmodel1.summary())
print(lmodel1.summary())

print("예측하기 ---")
x_new = pd.DataFrame({'tv': advdf.tv[:3]})
print("실제값: ", advdf.sales[:3])
print("예측값: ", lmodel1.predict(x_new))
print("새 자료로 예측하기 ---")
x_new = pd.DataFrame({'tv': [100, 200, 300]})
print("새 자료 예측값: ", lmodel1.predict(x_new).values) # [11.78625759 16.53992164 21.29358568]

# 시각화
plt.scatter(advdf.tv, advdf.sales)
y_pred = lmodel1.predict(advdf.tv)
plt.plot(advdf.tv, y_pred, c = 'red')
plt.xlabel('tv')
plt.ylabel('sales')
# plt.show()
plt.close()

print('*** 선형회귀분석의 기존 가정 충족 조건 ***')
# 잔차(예측값)항 구하기
fitted = lmodel1.predict(advdf)
# print(fitted)
residual = advdf['sales'] - fitted
print(residual)
print('실제값: ', advdf['sales'])
print('예측값: ', fitted[:5].values)
print('잔차값: ', residual[:5].values)
print('잔차의 평균값: ', np.mean(residual)) # -1.4210854715202005e-15

print('정규성: 잔차가 정규성을 따르는지 확인', )
stat, pv = shapiro(residual)

print(f'Shapiro-Wilk Test => 통계량: {stat:.4f}, p-value:{pv:.4f}')
 # Shapiro-Wilk Test => 통계량: 0.9905, p-value:0.2133 > 0.05
print('정규성 만족' if pv > 0.05 else '정규성 위배 가능성')
sm.qqplot(residual, line='s')
plt.title('잔차 Q-Q plot')
# plt.show()
plt.close() # 정규성 만족이나 커브를 그려가나는 부분이 좋지 않다.

print('2)선형성 : 독립변수(feature)의 변화에 따라 종속변수도 일정 크기로 변화해야 한다.')

reset_result = linear_reset(lmodel1, power=2, use_f=True)
print(f'linear_reset test : F={reset_result.fvalue:.4f}, p={reset_result.pvalue:.4f}')
print('선형성 만족' if reset_result.pvalue > 0.05 else '선형성 위배 가능성')

# 시각화로 확인
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
# plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='gray')
# plt.show()
plt.close()

print("3)독립성 : 독립변수의 값이 서로 관련되지 않아야 한다.(여성 데이터를 이용하여 유전 정보를 조사 -> 여성 데이터의 여성이 서로 가족이면 안 됌)")
# 독립성 가정은 잔차 간에 자기상관이 없어야 한다.
# 자기상관: 회귀분석 등에서 관측된 값과 추정된 값의 차이인 잔차들이 서로 연관되어 있는 상태
# 듀빈-왓슨(Durbin-Watson) 검정으로 확인
print(lmodel1.summary())
# Durbin-Watson: 1.935 -> 2에 근사하면 자기상관 없음
# | ----------2-----------|
# 0                       4
# 0으로 가면 음의 자기상관, 4로 가면 양의 자기상관

# 참고: Cook's distance
# 하나의 관측치가 전체 모델에 얼마나 영향을 주는 지 수치화한 지표
cd, _ = OLSInfluence(lmodel1).cooks_distance # 쿡의 거리 값과 인덱스

# 쿡 값 중 가장 큰 5개 관측치 확인
print(cd.sort_values(ascending=False).head())

# 인덱스 번째에 해당되는 원본 자료 확인
print(advdf.iloc[[35, 178, 25, 175, 131]])
#         tv  radio  newspaper  sales
# 35   290.7    4.1        8.5   12.8
# 178  276.7    2.3       23.7   11.8
# 25   262.9    3.5       19.5   12.0
# 175  276.9   48.9       41.8   27.0
# 131  265.2    2.9       43.0   12.7
# 해석: 대체적으로 tv 광고비는 높은데, 그에 반해 sales가 적음 - 모델이 예측하기 어려운 포인트들

# 시각화(Cook's distance 시각화)
fig = sm.graphics.influence_plot(lmodel1, alpha=0.05, criterion='cooks')
# plt.xlabel('leverage')
plt.show()
plt.close()