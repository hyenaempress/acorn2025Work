# 날씨, 비가 온다 안온다
# 날씨 예보(강우 여부)

# 비가오는 데에 영향없는 칼럼 뺌 (Date,RainToday) 
# RainTomorrow 비가온다 1 안온다 0 

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split 
import statsmodels.api as sm 
import statsmodels.formula.api as smf
# statsmodels.formula = 빈 폴더 주소만 불러온 것
# statsmodels.formula.api = 폴더 안에 있는 실행파일(ols, glm 등)을 바로 꺼내 쓸 수 있게 해주는 것

data=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv')
print('\ndata.head(3),data.shape:\n',data.head(3),data.shape)
# data.head(3),data.shape:
#           Date  MinTemp  MaxTemp  Rainfall  Sunshine  ...  Pressure  Cloud  Temp  RainToday  RainTomorrow
# 0  2016-11-01      8.0     24.3       0.0       6.3  ...    1015.0      7  23.6         No           Yes
# 1  2016-11-02     14.0     26.9       3.6       9.7  ...    1008.4      3  25.7        Yes           Yes
# 2  2016-11-03     13.7     23.4       3.6       3.3  ...    1007.2      7  20.2        Yes           Yes
# [3 rows x 12 columns] (366, 12) '

#의미없는 데이터 뺌
data2=pd.DataFrame()
data2=data.drop(['Date','RainToday'],axis=1)
data2['RainTomorrow']=data2['RainTomorrow'].map({'Yes':1,'No':0}) 
print('\ndata2.head(3),data2.shape:\n',data2.head(3),data2.shape)
#data2.head(3),data2.shape:
#     MinTemp  MaxTemp  Rainfall  Sunshine  WindSpeed  Humidity  Pressure  Cloud  Temp  RainTomorrow
# 0      8.0     24.3       0.0       6.3         20        29    1015.0      7  23.6             1
# 1     14.0     26.9       3.6       9.7         17        36    1008.4      3  25.7             1
# 2     13.7     23.4       3.6       3.3          6        69    1007.2      7  20.2             1 (366, 10)

print('\ndata2.RainTomorrow.unique():\n',data2.RainTomorrow.unique())
# data2.RainTomorrow.unique():
#  [1 0] 

# 학습데이터와 검정데이터로 분리  
# 학습데이터로 검정하면 오버피팅 가능성 있음 
train,test=train_test_split(data2,test_size=0.3,random_state=42) # 7:3 테스트가 3  
print('train.shape,test.shape:\n',train.shape,test.shape)
# train.shape,test.shape:
#  (256, 10) (110, 10) 

# 분류모델 만들기 
print('\ndata2.columns:\n',data2.columns)
# data2.columns:
#  Index(['MinTemp', 'MaxTemp', 'Rainfall', 'Sunshine', 'WindSpeed', 'Humidity',
#        'Pressure', 'Cloud', 'Temp', 'RainTomorrow'],
#       dtype='object') 
# 종속변수 RainTomorrow 

col_select='+'.join(train.columns.difference(['RainTomorrow'])) 
print('\ncol_select:\n',col_select) 
# col_select:
#  Cloud+Humidity+MaxTemp+MinTemp+Pressure+Rainfall+Sunshine+Temp+WindSpeed
my_formula='RainTomorrow~'+col_select 
#model=smf.glm(formula=my_formula,data=train,family=sm.families.Binomial()).fit() # fit 최소제곱법써라
#혼란표 사용하려고
model=smf.logit(formula=my_formula,data=train).fit() 
print('\nmodel.summary():\n',model.summary()) 
# model.summary():
#                   Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:           RainTomorrow   No. Observations:                  253
# Model:                            GLM   Df Residuals:                      243
# Model Family:                Binomial   Df Model:                            9
# Link Function:                  Logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -72.927
# Date:                Thu, 28 Aug 2025   Deviance:                       145.85
# Time:                        11:56:23   Pearson chi2:                     194.
# No. Iterations:                     6   Pseudo R-squ. (CS):             0.3186
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    219.3889     53.366      4.111      0.000     114.794     323.984
# Cloud          0.0616      0.118      0.523      0.601      -0.169       0.293
# Humidity       0.0554      0.028      1.966      0.049       0.000       0.111
# MaxTemp        0.1746      0.269      0.649      0.516      -0.353       0.702
# MinTemp       -0.1360      0.077     -1.758      0.079      -0.288       0.016
# Pressure      -0.2216      0.052     -4.276      0.000      -0.323      -0.120
# Rainfall      -0.1362      0.078     -1.737      0.082      -0.290       0.018
# Sunshine      -0.3197      0.117     -2.727      0.006      -0.550      -0.090
# Temp           0.0428      0.272      0.157      0.875      -0.489       0.575
# WindSpeed      0.0038      0.032      0.119      0.906      -0.059       0.066
# ==============================================================================
# P>|z| 에서 0.05보다 큰 변수들... 그냥 넘어감 우선
# 컬럼들 하나씩 빼야한다... 이유는 다른 연계된게 있을수 있으므로 한번에 다빼지X 
print('\nmodel.params:\n',model.params) 
# model.params:
#  Intercept    219.388868
# Cloud          0.061599
# Humidity       0.055433
# MaxTemp        0.174591
# MinTemp       -0.136011
# Pressure      -0.221634
# Rainfall      -0.136161
# Sunshine      -0.319738
# Temp           0.042755
# WindSpeed      0.003785
# dtype: float64 

# 예측값 구하기
print('\n예측값:\n',np.rint(model.predict(test)[:5])) 
print('\n 실제값 :\n',test['RainTomorrow'][:5].values)  
# 예측값:
#  193    0.0
# 33     0.0
# 15     0.0
# 310    0.0
# 57     0.0
# dtype: float64

#  실제값 :
#  [0 0 0 0 0] 

# 분류 정확도 확인 
# glm 은 pred_table 지원X (logit 에서는 가능)
# 라인 63 참고
conf_tab=model.pred_table() 
from sklearn.metrics import accuracy_score 
pred=model.predict(test) 
print('\n 혼란표 conf_tab :\n',conf_tab)  
#  혼란표 conf_tab :
#  [[197.   9.]
#  [ 21.  26.]]
print('\n 분류 정확도 :\n',(conf_tab[0][0]+conf_tab[1][1])/len(train) )  
#  분류 정확도 :
#  0.87109375
print('\n 분류 정확도 accuracy_score :\n',accuracy_score(test['RainTomorrow'],np.around(pred)))  
#  분류 정확도 accuracy_score :
#  0.8727272727272727 

