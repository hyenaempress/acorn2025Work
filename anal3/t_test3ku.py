# 비(눈) 여부(두 개의 집단)에 따른 매출액에 따른 평균 차이 검정
# 공통 칼럼이 년월일인 두 개의 파일을 조합을 해서 작업
# 귀무가설: 강수량에 따른 음식점 매출액 평균 차이는 없다.
# 대립가설: 강수량에 따른 음식점 매출액 평균 차이는 없다.

# 강수량에 따른 음식점 매출액의 평균 차이 검정
import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 자료 읽기
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(3)) # 328 entries, 3 columns / YMD: 20190514
print(sales_data.info())

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(3)) # 702 entries, 9 columns / tm:2018-06-01 
print(wt_data.info())

# sale에 매출액이 있으므로 데이터의 날짜를 기준으로 두개의 자료, 병합 작업 진행
wt_data.tm = wt_data.tm.map(lambda x: x.replace('-', ''))
print(wt_data.head(3)) # tm: 20180601

# 두 데이터 병합
frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head(3), ' ', len(frame))
print(frame.columns) # ['YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes']
data = frame.iloc[:, [0, 1, 7, 8]] # 날짜, 매출액, 최고기온, 강수량
print(data.head(3))
print(data.isnull().sum())
print('강수 여부에 따른 매출액 평균 차이가 유의미한지 확인하기')
# data['rain_yn'] = (data['sumRn'] > 0).astype(int) # 비옴: 1, 안옴: 0
data['rain_yn'] = (data.loc[:, ('sumRn')] > 0) * 1
print(data.head(5))

sp = np.array(data.iloc[:, [1, 4]]) # AMT와 rain_yn
tg1 = sp[sp[:, 1] == 0, 0] # 집단1 : 비 안올 때 매출액
tg2 = sp[sp[:, 1] == 1, 0] # 집단2 : 비 올 때 매출액
print('tg1', tg1[:3])
print('tg2', tg2[:3])

import matplotlib.pyplot as plt
plt.boxplot([tg1, tg2], meanline=True, showmeans=True, notch=True)
plt.show()

print('두 집단 평균: ', np.mean(tg1), ' vs. ', np.mean(tg2))
# 761040  vs.  757331

# 정규성 검정
print(len(tg1), ' ', len(tg2))
print('tg1: pvalue:', stats.shapiro(tg1).pvalue) # tg1: pvalue: 0.056 > 0.05 정규성 만족
print('tg2: pvalue:', stats.shapiro(tg2).pvalue) # tg2: pvalue: 0.882 > 0.05 정규성 만족

# 등분산성
print('등분산성 : ', stats.levene(tg1, tg2).pvalue) # 등분산성 :  0.7123452333011173

print(stats.ttest_ind(tg1, tg2, equal_var=True)) # statistic=0.101, pvalue=0.919, df=326.0
# pvalue: 0.919 > 0.05 이므로 귀무가설 채택

# 결론: 강수 여부에 따른 매출액 평균은 차이가 없다.