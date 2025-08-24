# t-test 3 파일에서 했던 날씨 데이터를 또 한번 써보는거야
# 온도(서로 다른 세 개의 집단)에 따른 음식점 매출액의 평균 차이 검정
# 공통 칼럼(columns)이 연월일인 두 개의 파일을 조합하여 작업

# 귀무가설(H0) : 온도에 따른 음식점 매출액 평균의 차이는 없다.
# 대립가설(H1) : 온도에 따른 음식점 매출액 평균은 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

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

print(data.maxTa.describe())
# 일별 최고 온도(연속형) 변수를 이용해 명목형(구간화) 변수 추가
data['ta_gubun'] = pd.cut(data.maxTa, bins = [-5, 8, 24, 37], labels = [0,1,2])
print(data)
print(data.ta_gubun.unique())
print(data.isnull().sum())

# 최고 온도를 세 그룹으로 나눈 뒤, 등분산/정규성 검정
x1 = np.array(data[data.ta_gubun == 0].AMT)
x2 = np.array(data[data.ta_gubun == 1].AMT)
x3 = np.array(data[data.ta_gubun == 2].AMT)
print('x1[:5] : ',x1[:5], 'len(x1) : ', len(x1))
print('-' * 100)
print(stats.levene(x1,x2,x3))
# LeveneResult(statistic=np.float64(3.276731509044197), 
# pvalue=0.03900        <---- 등분산성을 만족하지 못했다.
print(stats.shapiro(x1).pvalue)     # p-value = 0.2481924204382751
print(stats.shapiro(x2).pvalue)     # p-value = 0.03882572120522948     # 정규성 만족 X
print(stats.shapiro(x3).pvalue)     # p-value = 0.3182989573650957      # 하지만 대체로 만족

spp = data.loc[:,['AMT', 'ta_gubun']]
print(spp.groupby('ta_gubun').mean())

print(pd.pivot_table(spp, index = ['ta_gubun'], aggfunc = 'mean'))

# ANOVA 진행
sp = np.array(spp)
group1 = sp[sp[:,1] == 0, 0]
group2 = sp[sp[:,1] == 1, 0]
group3 = sp[sp[:,1] == 2, 0]

print(stats.f_oneway(group1, group2, group3))
print('-' * 100)
# F_onewayResult
# (statistic=np.float64(99.1908012029983), 
# pvalue=np.float64(2.360737101089604e-34)) <<<<<<<<<< 0.05
# 따라서 귀무기각
# 그래서 내릴 결론은, 강수량(눈,비)에는 매출액 차이가 없엇지만 
# 온도에는 큰 영향이 잇다.
# 음식점의 매출액은 온도에 영향을 받는다.(H1)

# 참고 : 등분산성 만족 X : Welch's
# pip install pingouin     # 라이브러리 하나 또 깔앗다 깐거 참 많네
from pingouin import welch_anova
print('welch_anova : \n', welch_anova(dv = 'AMT', between = 'ta_gubun', data = data))
print('-' * 100)
#      Source  ddof1     ddof2           F         p-unc       np2
# 0  ta_gubun      2  189.6514  122.221242  7.907874e-35  0.379038
# p-unc 가 p-value를 표현한 것 같애 
# p-value = 7.907874e-35 <<<<< 0.05 정규성 만족 못해

# 참고 : 정규성 만족 X : Kruskal wallis test
 
print(stats.kruskal(group1, group2, group3))
# KruskalResult
# (statistic=np.float64(132.7022591443371), 
# pvalue=np.float64(1.5278142583114522e-29))        0.05에 비해 엄청 작다 따라서 귀무기각
# 그얘기는 H1 채택 : 온도에 따라 매출액에 차이가 발생한다.

# 사후분석
# 이건 ANOVA 2번 파일에서 가져오자

# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult = pairwise_tukeyhsd(endog = spp['AMT'], groups = spp['ta_gubun'])
print(turkyResult)

# =================================================================
# group1 group2   meandiff   p-adj    lower        upper     reject # True가 나왔다는건
# ----------------------------------------------------------------- # 온도와 매출이 독립적이지 않다는거야
#      0      1 -214255.4486   0.0  -296755.647 -131755.2503   True
#      0      2 -478651.3813   0.0 -561484.4539 -395818.3088   True
#      1      2 -264395.9327   0.0 -333326.1167 -195465.7488   True

turkyResult.plot_simultaneous(xlabel = 'mean', ylabel = 'group')
plt.show()
plt.close()





