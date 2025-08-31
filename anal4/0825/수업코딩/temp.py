# https://cafe.daum.net/flowlife/SBU0/44

# 회귀분석 문제 3)    
# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon 에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측. 
# ols 사용함
import pandas as pd 
import statsmodels.formula.api as smf 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
plt.rc('font',family='malgun gothic') 

df=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Carseats.csv')
print(df.info()) 
df=df.drop([df.columns[6],df.columns[9],df.columns[10]],axis=1)
print(df.corr())
#종속 sales, 독립 Income 
lmodel=smf.ols(formula='Sales~Income+Advertising+Price+Age',data=df).fit()
print('요약결과:',lmodel.summary())

# 요약결과:                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  Sales   R-squared:                       0.371
# Model:                            OLS   Adj. R-squared:                  0.364 >0.15보다 크면 됨
# Method:                 Least Squares   F-statistic:                     58.21
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           1.33e-38 < 0.05 만족
# Time:                        16:49:36   Log-Likelihood:                -889.67
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
# Omnibus:                        3.285   Durbin-Watson:                   1.931 -> 2에 근사
# Prob(Omnibus):                  0.194   Jarque-Bera (JB):                3.336
# Skew:                           0.218   Prob(JB):                        0.189
# Kurtosis:                       2.903   Cond. No.                     1.01e+03
# ==============================================================================

# 독립변수 모두 0.05보다 작다  
# -----------------작성된 모델 저장 후 읽어서 사용함 -----------------
# ---------------방법 1 
# pickle모듈 사용 
#import pickle 
# 저장
# with open('lm9_Carseats.pickle',mode='wb') as obj:
#     pickle.dump(lmodel,obj)
#읽기
# with open('lm9_Carseats.pickle',mode='rb') as obj:
#         mymodel1=pickle.load(obj) 
# mymodel1.predict('~~~~') 

# ---------------방법 2
# joblib 모듈 사용
#import joblib 
# # 저장
# joblib.dump(lmodel,'mymodel.model')
# # 읽기
# mymodel=joblib.load(lmodel,'mymodel.model')
# mymodel.predict('~~~')


# 선형회귀분석기본충족조건 
#독립변수만가진df
#print(df.head(3))
df_lm=df.iloc[:,[0,2,3,5,6]] 
# 잔차항 얻기 
fitted=lmodel.predict(df_lm) 
residual=df_lm['Sales']-fitted 
print(f'residual[:3]: {residual[:3]}') 
print(f'잔차의 평균: {np.mean(residual)}') 

import seaborn as sns
print('선형성: 잔차가 일정하게 분포되어야 함')
#시각화로 확인
sns.regplot(x=fitted,y=residual,lowess=True,line_kws={'color':'green'}) # 파선에 가깝게, 호를 그리지않으면 선형성 만족이다. 
plt.plot([fitted.min(),fitted.max()],[0,0],'--',color='gray')
plt.show() 
plt.close()   

import scipy.stats as stats
print('정규성: 잔차항이 정규 분포를 따라야 함')
sr=stats.zscore(residual)
(x,y),_=stats.probplot(sr)
sns.scatterplot(x=x,y=y) 
plt.plot([-3,3],[-3,3],'--',color="gray")
plt.show()
plt.close()    

print('샤피로테스트:',stats.shapiro(residual))
# 샤피로테스트: ShapiroResult(statistic=np.float64(0.9949221268962882), pvalue=np.float64(0.21270047355494331)) -> 0.212 정규성 만족

print('독립성: 독립변수의 값이 서로 관련되지않아야한다')
# 듀빈왓슨 방법1
# Durbin-Watson: 1.931 -> 2에 근사함 자기상관없다 

import statsmodels.api as sm
# 듀빈왓슨 방법2 
print(f'듀빈왓슨2: {sm.stats.stattools.durbin_watson(residual)}') 
# print()

# -------------------------ols 내용 끝-------------------------
# print('등분산성: ')
# print('다중공선성: ')


