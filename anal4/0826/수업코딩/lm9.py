# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음)
# Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import shapiro
import statsmodels.api as sm # Quantiel-Quantile plot 지원
from statsmodels.stats.diagnostic import linear_reset # 모형 적합성
from statsmodels.stats.outliers_influence import OLSInfluence
import scipy.stats as stats
import joblib
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 데이터 로딩
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv")
print(df)
print(df.shape)
print(df.info())
df = df.drop(df.columns[[6, 9, 10]], axis=1)
print(df.corr())

lmodel = smf.ols(formula='Sales ~ Income + Advertising + Price + Age', data=df).fit()

print('요약결과: ', lmodel.summary())
# 요약결과:                     OLS Regression Results
# ==============================================================================
# Dep. Variable:                  Sales   R-squared:                       0.371
# Model:                            OLS   Adj. R-squared:                  0.364
# Method:                 Least Squares   F-statistic:                     58.21
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.33e-38
# Time:                        16:51:46   Log-Likelihood:                -889.67
# No. Observations:                 400   AIC:                             1789.
# Df Residuals:                     395   BIC:                             1809.
# Df Model:                           4
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# Intercept      15.1829      0.777     19.542      0.000      13.656      16.710
# Income          0.0108      0.004      2.664      0.008       0.003       0.019
# Advertising     0.1203      0.017      7.078      0.000       0.087       0.154
# Price          -0.0573      0.005    -11.932      0.000      -0.067      -0.048
# Age            -0.0486      0.007     -6.956      0.000      -0.062      -0.035
# ==============================================================================
# Omnibus:                        3.285   Durbin-Watson:                   1.931
# Prob(Omnibus):                  0.194   Jarque-Bera (JB):                3.336
# Skew:                           0.218   Prob(JB):                        0.189
# Kurtosis:                       2.903   Cond. No.                     1.01e+03
# ==============================================================================

# Income, Advertising, Price, Age 모두 < 0.05
# 작성된 모델 저장 후 읽어서 사용하기
# pickle 모듈을 사용
# import pickle
# # 저장
# with open('model.pickle', mode='wb') as obj:
#     pickle.dump(lmodel, obj)

# # 읽기
# with open('model.pickle', mode='rb') as obj:
#     mymodel = pickle.load(obj)
# mymodel.predict('~~~')

# joblib 모듈 사용
# joblib.dump(lmodel, './01_big_data_machine_learning/data/mymodel.model')

# *** 선형회귀분석의 기본 충족 조건 ***
df_lm = df.iloc[:, [0, 2, 3, 5, 6]] # Income, Advertising, Price, Age

# 잔차
fitted = lmodel.predict(df_lm)
print("fitted: ", fitted)
residual = df_lm['Sales'] - fitted
print(residual[:3])
print('잔차의 평균: ', np.mean(residual))

print('\n선형성 : 잔차가 일정하게 분포되어야 함')
sns.regplot(x=fitted, y = residual, lowess=True, line_kws={'color': 'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='blue')
# plt.show() # 선형성 만족
plt.close()

# 정규성
print('\n정규성 : 잔차항이 정규 분포를 따라야 함')
sr = stats.zscore(residual)
(x, y), _ = stats.probplot(sr)
sns.scatterplot(x=x, y=y)
plt.plot([-3, 3], [-3, 3], '--', color='gray')
# plt.show()
plt.close()
print('shapiro test: ', stats.shapiro(residual)) # (0.994922126896289, 0.21270047355504762) > 0.05 만족

print('\n 독립성: 독립 변수의 값이 서로 관련되지 않아야 한다.')
# 듀빈 왓슨 검정으로 확인(Durbin-Watson: 1.931)
# 2에 근사하므로 자기 상관이 없다.

# 시각화(Cook's distance 시각화)
print('Durbin-Watson: ', sm.stats.durbin_watson(lmodel.resid))
# Durbin-Watson:  1.9314981270829588 -- 2d에 근사하면 자기 상관 없음

print("\n등분산성(homoscedasticity) : ")
# 등분산성: 그룹 간의 분산이 유사해야 한다. 독립 변수의 모든 값에 대해 잔차의 분산이 동일해야 함
# 시각화(z-score)
sr = stats.zscore(residual)
sns.regplot(x=fitted, y=np.sqrt(abs(sr)), lowess=True, line_kws={'color': 'red'})
# plt.show()
plt.close()

bp_test = het_breuschpagan(residual, lmodel.model.exog)
print(f'통계량: {bp_test[0]:.4f}, p-value : {bp_test[1]:.4f}') # 통계량: 1.1277, p-value : 0.8899 > 0.05 등분산성 만족

print("\n다중공산성(multicollinearity) : ")
# 다중공산성: 다중회귀 분석 시 두 개 이상의 독립 변수가 서로 상관관계를 가질 때 발생
# 분산팽창지수(VIF : Variance Inflation Factor) : 연속형의 경우 10을 넘으면 의심
imsidf = df[['Income', 'Advertising', 'Price', 'Age']]
vifdf = pd.DataFrame()
vifdf['vif_value'] = [variance_inflation_factor(imsidf.values, i) for i in range(imsidf.shape[1])]
print(vifdf) # 모든 독립변수가 VIF 값 10 미만이므로 다중공산성 문제 없음
# 다중공산성(multicollinearity) :
#    vif_value
# 0   5.971040
# 1   1.993726
# 2   9.979281
# 3   8.267760

# 저장된 모델을 읽어 새로운 데이터에 대한 예측

# 읽기
ourmodel = joblib.load('./01_big_data_machine_learning/data/mymodel.model')

new_df = pd.DataFrame({
    'Income': [35, 62, 25],
    'Advertising': [6, 3, 11], 
    'Price': [105, 88, 77],
    'Age': [33, 55, 22]
})

new_pred = ourmodel.predict(new_df)
print("Sales 예측 결과: \n", new_pred)


