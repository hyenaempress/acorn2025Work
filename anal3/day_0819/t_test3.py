# 비(눈) 여부(서로 다른 두 개의 집단)에 따른 음식점 매출액의 평균 차이 검정
# 공통 칼럼(columns)이 연월일인 두 개의 파일을 조합하여 작업

# 귀무가설(H0) : 강수량에 따른 음식점 매출액 평균의 차이는 없다.
# 대립가설(H1) : 강수량에 따른 음식점 매출액 평균은 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 자료 읽기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv',dtype = {'YMD' : 'object'})
print(sales_data.head(3))   # 328 rows, 3 columns
print(sales_data.info())



# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(3)) # 702 rows, 9 columns   
print(wt_data.info())
print('-' * 100)

# sales 기준으로 데이터의 날짜를 기준으로 두 개의 자료를 병합 작업 진행
wt_data.tm = wt_data.tm.map(lambda x:x.replace('-',''))
print(wt_data.head(3)) 
print('-' * 100)

frame = sales_data.merge(wt_data, how = 'left', left_on = 'YMD', right_on='tm')
print(frame.head(3), ' ', len(frame))
print(frame.columns)
print('-' * 100)
# YMD    AMT  CNT  stnId        
# tm  avgTa  minTa  maxTa  sumRn  maxWs  avgWs  ddMes

data = frame.iloc[:, [0,1,7,8]]     # 날짜, 매출액, 최고기온, 강수량
print(data.head(3))
print('-' * 100)
print(data.isnull().sum())
print('-' * 100)
print('강수 여부에 따른 매출액 평균 차이가 유의미한지 확인하기')
# data['rain_yn'] = (data['sumRn'] > 0).astype(int) # 비옴 : 1, 비 안옴 : 0
data['rain_yn'] = (data.loc[:,('sumRn')] > 0) * 1
print(data.head())

sp = np.array(data.iloc[:, [1,4]])  # AMT, rain_yn
tg1 = sp[sp[:, 1] == 0, 0]  # 집단 1 : 비 안올때 매출액
tg2 = sp[sp[:, 1] == 1, 0]  # 집단 2 : 비 올때 매출액

print('tg1 : ', tg1[:3])
print('tg2 : ', tg2[:3])

import matplotlib.pyplot as plt
plt.boxplot([tg1,tg2], meanline = True, showmeans=True, notch = True)
# plt.show()

print('두 집단 평균 : ', np.mean(tg1), ' v.s ', np.mean(tg2))
# 두 집단 평균 :  761040.2  v.s  757331.5
# 이 평균의 차이를 의미가 있는지 없는지 검정해야한다

# 정규성 검정 (shapiro를 쓰나?)
print(len(tg1), ' ', len(tg2))
        # pvalue : 0.056050644029515644 - 정규성 만족
print(stats.shapiro(tg2).pvalue)        # pvalue : 0.8827503155277691 - 정규성 만족

# 등분산성 검정
print('등분산성 : ', stats.levene(tg1, tg2).pvalue) 
# 등분산성 :  0.7123452333011173 이므로 유의수준 0.05보다 높다
# 등분산성이 성립함으로 가정할 수 있다.
# 그러니 밑의 ttest_ind에서 equal_var를 트루로 설정할 수 있다.

print(stats.ttest_ind(tg1, tg2, equal_var = True))
# TtestResult(statistic=np.float64(0.10109828602924716), 
# pvalue=np.float64(0.919534587722196), df=np.float64(326.0))
# pvalue는 0.91 이상이므로 유의수준 0.05보다 크다
# 따라서 귀무가설이 채택된다
# 귀무가설(H0) : 강수량에 따른 음식점 매출액 평균의 차이는 없다.
# 강수 여부에 따른 매출액 평균은 변동이 없다.