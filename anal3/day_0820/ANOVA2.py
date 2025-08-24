# 일원분산분석 연습
# 강남구에 있는 GS 편의점 3개 지역 알바생의 급여에 대한평균의 차이가 있는가?
# 귀무 : GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 없다.
# 대립 : GS 편의점 3개 지역 알바생의 급여에 대한 평균의 차이가 있다.

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt'
# data1 = pd.read_csv(url, header = None)
# print(data1.values)
data = np.genfromtxt(url, delimiter=',')
print(data, type(data), data.shape)     # ndarray (22, 2)

# 3개 집단의 월급, 평균 얻기
gr1 = data[data[:,1] == 1, 0] # 0번째 열의 월급 가져와
gr2 = data[data[:,1] == 2, 0] # 0번째 열의 월급 가져와
gr3 = data[data[:,1] == 3, 0] # 0번째 열의 월급 가져와
print(gr1, ' ', np.mean(gr1))   # 316.625
print(gr2, ' ', np.mean(gr2))   # 256.4
print(gr3, ' ', np.mean(gr3))   # 278.0

# 정규성
print(stats.shapiro(gr1).pvalue)    # 0.3336828974377483
print(stats.shapiro(gr2).pvalue)    # 0.6561053962402779
print(stats.shapiro(gr3).pvalue)    # 0.8324811457153043
# 3 데이터 모두 정규성 만족

# 등분산성
print(stats.levene(gr1, gr2,gr3).pvalue)    # 0.045846812634186246
print(stats.bartlett(gr1, gr2,gr3).pvalue)  # 0.3508032640105389

# plt.boxplot([gr1, gr2, gr3], showmeans = True)
# plt.show()
# plt.close()

# ANOVA 검정 방법 1 : anova_lm
df = pd.DataFrame(data, columns = ['pay','group'])
print(df)
lmodel = ols('pay ~ C(group)', data = df).fit()
# ols -> ordinary least square, .fit 이건 학습시키는 구문이래
print(anova_lm(lmodel, type = 2))   # p-value 0.043589

#             df        sum_sq      mean_sq         F    PR(>F)
# C(group)   2.0  15515.766414  7757.883207  3.711336  0.043589
# Residual  19.0  39716.097222  2090.320906       NaN       NaN

# ANOVA 검정 방법 2 : f_oneway
f_statistic, p_value = stats.f_oneway(gr1, gr2, gr3)
print('f_statistic : ', f_statistic)    # f_statistic :  3.71133
print('p_value : ', p_value)            # p_value :  0.0435 < 0.05 그래서 귀무기각


# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkyResult = pairwise_tukeyhsd(endog = df.pay, groups = df.group)
print(turkyResult)

turkyResult.plot_simultaneous(xlabel = 'mean', ylabel = 'group')
plt.show()
plt.close()

