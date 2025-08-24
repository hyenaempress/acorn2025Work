import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot

# 한글 깨짐 방지
plt.rc('font', family='Malgun Gothic')

# =========================
# 실습 1) 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향
# =========================
url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt'
data = pd.read_csv(url)
print(data.head(3), data.shape)
print("태아수:", data['태아수'].unique())
print("관측자수:", data['관측자수'].unique())

# 교호작용 포함 ANOVA
reg = ols('머리둘레 ~ C(태아수) * C(관측자수)', data=data).fit()
result = anova_lm(reg, type=2)
print(result)
print('-'*100)

# =========================
# 실습 2) poison 종류와 treat가 독 퍼짐 시간 평균에 영향
# =========================
data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv', index_col=0)
print(data2.head(3), data2.shape)

# 균형 확인
print(data2.groupby('poison').agg(len))
print('-'*100)
print(data2.groupby('treat').agg(len))
print('-'*100)
print(data2.groupby(['poison', 'treat']).agg(len))
print('-'*100)

# 교호작용 포함 ANOVA
result2 = ols('time ~ C(poison) * C(treat)', data=data2).fit()
anova_result2 = anova_lm(result2)
print(anova_result2)
print('-'*100)

# =========================
# 사후분석 (Tukey HSD)
# =========================
tkResult1 = pairwise_tukeyhsd(endog=data2.time, groups=data2.poison)
print(tkResult1)
print('-'*100)

tkResult2 = pairwise_tukeyhsd(endog=data2.time, groups=data2.treat)
print(tkResult2)
print('-'*100)

# 시각화
tkResult1.plot_simultaneous(xlabel='mean', ylabel='poison')
tkResult2.plot_simultaneous(xlabel='mean', ylabel='treat')

# =========================
# 교호작용 시각화
# =========================
# 범주형으로 변환
data2['poison'] = data2['poison'].astype(str)
data2['treat'] = data2['treat'].astype(str)

plt.figure(figsize=(8,6))
interaction_plot(data2['poison'], data2['treat'], data2['time'],
                 colors=['red','blue','green','orange'], markers=['o','s','D','^'])
plt.xlabel('독 종류')
plt.ylabel('퍼지는 시간')
plt.title('독 종류와 처치 방법 간 교호작용 확인')
plt.grid(True)
plt.show()
