# 이원분산 분석 : 두 개의 요인에 대한 집단(독립변수) 각각이 종속변수의 평균에 영향을 주는 경우

# 가설이 주효과 2개, 교호작용 1개가 나옴 (그래서 교호작용이 뭐야?)
# 교호작용(interation term) : 한 쪽 요인이 취하는 수준에 따라 다른 쪽 요인이
# 영향을 받는 요인의 조합효과를 말하는 것으로 상승과 상쇄효과가 있다.
# 예) 초밥과 간장(상승효과), 감자튀김과 간장, 초밥과 케첩(이건 상쇄효과) . . .


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot 
# interaction plot 이건 지피티한테 교호작용 여부 보고싶다그래서 코드 받아본거야 수업내용은 아니엇어

plt.rc('font', family = 'Malgun Gothic') # 한글 깨짐방지

# 실습 1) 태아 수와 관측자 수가 태아의 머리둘레 평균에 영향을 주는가? 
# (그럼 관측자 수랑 태아 머리둘레는 상쇄효과인가? 머지)

# 주효과 가설

# 귀무가설(H0) : 태아 수와 태아의 머리둘레 평균은 차이가 없다.
# 대립가설(H1) : 태아 수와 태아의 머리둘레 평균은 차이가 있다.

# 귀무가설(H0) : 태아 수와 관측자 수의 머리둘레 평균은 차이가 없다.
# 대립가설(H1) : 태아 수와 관측자 수의 머리둘레 평균은 차이가 있다.

# 교호작용 가설
# 귀무가설(H0) : 교효작용이 없다. (태아수와 간측자수는 관련이 없다.)
# 대립가설(H1) : 교효작용이 있다. (태아수와 간측자수는 관련이 있다.)

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt'
data = pd.read_csv(url)
print(data.head(3), data.shape)        # (36, 3)
print(data['태아수'].unique())          # [1 2 3]
print(data['관측자수'].unique())         # [1 2 3 4]

# reg = ols('머리둘레 ~ C(태아수) + C(관측자수)', data = data).fit()  
# 이렇게 복잡하게 꼬아놓으면 교호작용은 확인할 수 없대

# reg = ols('머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)', data = data).fit()  
reg = ols('머리둘레 ~ C(태아수) * C(관측자수)', data = data).fit()  
# 이렇게 하면 교효작용 확인할 수 있대
result = anova_lm(reg, type = 2)
print(result)
print('-' * 100)
#                   df      sum_sq     mean_sq            F        PR(>F) <-이게 피밸류
# C(태아수)           2.0  324.008889  162.004444  2113.101449  1.051039e-27       < 0.05 귀무기각
# C(관측자수)          3.0    1.198611    0.399537     5.211353  6.497055e-03      > 0.05 귀무채택
# C(태아수):C(관측자수)   6.0    0.562222    0.093704     1.222222  3.295509e-01    < 0.05 귀무기각
# Residual        24.0    1.840000    0.076667          NaN           NaN
# 태아 수는 머리둘레에 강력한 영향을 미침. 관측자 수는 유의한 영향을 미친다.
# 하지만 태아 수와 관측자 수의 상호관계는 유의하지 않다.

# 실습 2) poison 종류와 treat가 독퍼짐 시간의 평균에 영향을 주는가?
# 주효과 가설
# 귀무가설(H0) : poison 종류와 treat가 독퍼짐 시간의 평균에 차이가 없다.
# 대립가설(H1) : poison 종류와 treat가 독퍼짐 시간의 평균에 차이가 있다.

# 교호작용 가설
# 귀무가설(H0) : 교효작용이 없다. (poison 종류와 treat(응급처치) 방법은 관련이 없다)
# 대립가설(H1) : 교효작용이 있다. (poison 종류와 treat(응급처치) 방법은 관련이 있다)

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv', index_col = 0)
print(data2.head(3),data2.shape)    # (48, 3)

# 데이터의 균형 확인
print(data2.groupby('poison').agg(len))
print('-' * 100)
print(data2.groupby('treat').agg(len))
print('-' * 100)
print(data2.groupby(['poison', 'treat']).agg(len))
print('-' * 100)
# 모든 집단별 표본 수가 동일하므로 균형설계가 잘 되어있다. 라고 할 수 있다.

result2 = ols('time ~ C(poison) * C(treat)', data = data2).fit()
print(anova_lm(result2))
print('-' * 100)
#                       df    sum_sq   mean_sq          F        PR(>F)
# C(poison)            2.0  1.033012  0.516506  23.221737  3.331440e-07 < 0.05 ----귀무 기각
# C(treat)             3.0  0.921206  0.307069  13.805582  3.777331e-06 < 0.05 ----귀무 기각
# C(poison):C(treat)   6.0  0.250138  0.041690   1.874333  1.122506e-01 > 0.05 이므로 상호 작용 효과는 없다
# 왜냐면 피밸류가 유의수준보다 크니까 귀무채택인데, 귀무가설은 독립적임을 가정했잖아

# C(poison) 독의 종류과 그 퍼짐 시간은 관계가 있어(귀무기각)
# C(treat) 치유제의 종류와 독의 퍼짐 시간은 관계가 있어(귀무기각)
# C(poison):C(treat) 하지만 독의 종류와 치유제의 종류는 퍼짐 시간에 대해 독립적이야(귀무채택)
# 그러니까 어떤 종류의 독이 들어와도 약 A만 써도 퍼짐 시간을 늦출 수 있다는 거지

# Residual            36.0  0.800725  0.022242        NaN           NaN

# 사후 분석(Post Hoc)
from statsmodels.stats.multicomp import pairwise_tukeyhsd   
# 이 라이브러리는 사후분석, 그니까 두 집단(독, 처치한 약)의 상관 관계를 알아보기 위함이야
tkResult1 = pairwise_tukeyhsd(endog = data2.time, groups = data2.poison)
print(tkResult1)
print('-' * 100)

# Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject      # 여기서 p-adj가 p-value를 나타낸거야
# ----------------------------------------------------
#      1      2  -0.0731 0.5882 -0.2525  0.1063  False      # p > 0.05  차이가 유의하지 않아(서로 관계 없어)
#      1      3  -0.3412 0.0001 -0.5206 -0.1619   True      # p < 0.05  차이가 유의해(서로 종속적이어보여)
#      2      3  -0.2681 0.0021 -0.4475 -0.0887   True      # p < 0.05  그래서 리젝트 당햇나봐
# ----------------------------------------------------

tkResult2 = pairwise_tukeyhsd(endog = data2.time, groups = data2.treat)
print(tkResult2)
print('-' * 100)
tkResult1.plot_simultaneous(xlabel = 'mean', ylabel = 'poison')
tkResult2.plot_simultaneous(xlabel = 'mean', ylabel = 'treat')

# 여기 밑에는 지피티한테 코드 달라그래서 본거야
# 근데 오류난다 그냥 나중에 쌤한테 물어보자

# plt.figure(figsize=(8,6))
# interaction_plot(data2['poison'], data2['treat'], data2['time'],
#                  colors=['red','blue','green'], markers=['o','s','D'])
# plt.xlabel('독 종류')
# plt.ylabel('퍼지는 시간')
# plt.title('독 종류와 처치 방법 간 교호작용 확인')
# plt.grid(True)

plt.show()
plt.close()