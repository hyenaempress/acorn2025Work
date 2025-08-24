# 세 개 이상의 모집단에 대한 가설검정–분산분석
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 
# 나누어진 각 집단 내의 분산으로 나누고 요인에 의한 분산이 의미있는 크기를
# 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 
# 제 1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 이용하게 된다.

# 분산의 성질과 원리를 이용하여, 평균의 차이를 분석한다.
# 즉, 평균을 직접 비교하지 않고 집단 내 분산과 집단 간 분산을 이용하여 집단의 평균이 서로
# 다른지 확인하는 방법이다.
# f-value = 그룹 간 분산(Between Variance) / 그룹 내 분산 (Within Variance) 

# f-value 값이 클 때, 그룹 내 분산값이 작을 때 데이터에 유의미성이 부여된다.

# * 서로 독립인 세 집단의 평균 차이 검정
# 실습 1) 세 가지 서로 다른 교육방법을 1개월동안 교육받은 교육생 80명을 대상으로 
# 실기시험을 실시하고 그 결과를 판단한다.

# 귀무가설(H0) : 세 가지 서로 다른 교육방법을 통한 학생들 실기시험의 평균 점수에 차이가 없다.
# 대립가설(H1) : 학생들의 실기시험 점수는 세 가지 서로 다른 교육방법에 달려있다.

# 독립변수 : 교육방법(세 가지 서로 다른 교육방법)
# 종속변수 : 시험 점수
# 일원분산분석(One Way ANOVA)

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv')
print(data.head())
print(data.shape)       # 80 x 4 의 데이터
print(data.describe())  # score의 의심 데이터 발견 (그니까 존내 튀는 이상한 점수들 438점같은거)

# 이상치를 차트로 확인
import matplotlib.pyplot as plt
# plt.hist(data.score)      # 그래프로 확 튀는 수치 보려고 한거야
# plt.boxplot(data.score)     # 확 튀는게 두개가 잇네
# plt.show()  
# plt.close()

# 이상치 제거
data = data.query('score <= 100')
print(len(data))    # 78

result = data[['method', 'score']]
print(result)
m1 = result[result['method'] == 1]
m2 = result[result['method'] == 2]
m3 = result[result['method'] == 3]
print(m1[:3])
print(m2[:3])
print(m3[:3])
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']

# 정규성
print('score1 : ', stats.shapiro(score1))   # pvalue=0.11558564512681252 > 0.05
print('score2 : ', stats.shapiro(score2))   # pvalue=0.3319001150712364 > 0.05
print('score3 : ', stats.shapiro(score3))   # pvalue=0.11558564512681252 > 0.05

print(stats.ks_2samp(score1,score2))
# KstestResult(statistic=np.float64(0.24725274725274726), 
# pvalue=np.float64(0.30968796298459966), 
# statistic_location=np.int64(59), 
# statistic_sign=np.int8(1))
# 두 집단의 동일 분포 여부 확인

# 등분산성(복수 집단 분산의 치우침 정도) 유의수준 0.05 이상이면 등분산성 성립
print('levene : ',stats.levene(score1,score2,score3))   # pvalue=0.1132285
print('levene : ',stats.fligner(score1,score2,score3))  # pvalue=0.10847
print('levene : ',stats.bartlett(score1,score2,score3)) # pvalue=0.105501

print('-' * 100)
# 교차표 등 작성 가능 ...
import statsmodels.api as sm

reg = ols("data['score'] ~ C(data['method'])", data = data).fit()

# 분산 분석표를 이용해 분산결과 작성

# 단일 회귀 모델 작성
table = sm.stats.anova_lm(reg,type = 2)
print(table)
#                      df        sum_sq     mean_sq         F    PR(>F)
# C(data['method'])   2.0     28.907967   14.453984  0.062312  0.939639
# Residual           75.0  17397.207418  231.962766       NaN       NaN

# p-value = 0.9396 > 0.05 이므로 귀무채택

# 사후 검정(Post Hoc Test)
# 분산 분석은 집단의 평균에 차이 여부만 알려줄 뿐, 각 집단 간의 평균 차이는 알려주지 않음.
# 각 집단 간의 평균 차이를 확인하기 위해 사후 검정 실시

from statsmodels.stats.multicomp import pairwise_tukeyhsd
turResult = pairwise_tukeyhsd(endog = data.score, groups = data.method)
print(turResult)

# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#      1      2   0.9725 0.9702 -8.9458 10.8909  False  # reject에 만약 True가 떳으면
#      1      3   1.4904 0.9363 -8.8183  11.799  False  # 데이터 유의미성이 의심받는거임
#      2      3   0.5179 0.9918 -9.6125 10.6483  False
# ----------------------------------------------------

turResult.plot_simultaneous(xlabel = 'mean', ylavel = 'group')
plt.show()
plt.close()