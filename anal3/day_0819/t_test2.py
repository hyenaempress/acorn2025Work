# 독립 표본 검정
# independent sample test
# 두 집단의 평균의 차이 
# 서로 다른 두 집단의 평균에 대한 통계 검정에 주로 사용됨
# 비교를 위해 평균과 표준편차 통계량을 사용한다.
# 평균값의 차이가 얼마인지, 표준편차는 얼마나 다른지 확해
# 분석 대상인 두 자료가 같을 가능성이 우연의 범위 5%에 들어가는지를
# 판별한다.
# 결국 t-test 는 두 집단의 평균과 표준편차 비율에 대한 대조 검정법이다.

#  * 서로독립인두집단의평균차이검정(independentsamplest-test)
# 남녀의 성적, A반과 B반의키, 경기도와 충청도의 소득 따위의 
# 서로 독립인 두 집단에서 얻은 표본을 독립표본(twosample)이라고한다.
# 실습)남녀 두 집단간 파이썬시험의 평균 차이 검정

# 귀무가설(H0) : 남녀 두 집단 간 파이썬 시험의 평균의 차이는 성별과 관계없다.
# 대립가설(H1) : 남녀 두 집단 간 파이썬 시험의 평균 차이는 성별과 관계있다.

# 95% 신뢰수준에서 우연히 발생할 확률이 5%보다 작느냐?
# 선행 조건 : 두 집단의 자료는 정규분포를 따른다. 분산이 동일하다(등분산성)
from scipy import stats
import pandas as pd
import numpy as np

male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
print(np.mean(male), ' ',np.mean(female))   # 83.8   72.24

two_sample = stats.ttest_ind(male,female)    # 이게 독립검정 t-test
print(two_sample)
# TtestResult(statistic=np.float64(1.233193127514512), 
# pvalue=np.float64(0.2525076844853278), df=np.float64(8.0))
# p-value값이 0.25 이상이므로 0.05보다 커 귀무 채택
# 해석 : 남녀 두 집단 간 파이썬 시험에 성별과 성적 간의 상관관계는 없다.

# 등분산 검정
print('\n등분산성 검정')
print('-' * 100)
from scipy.stats import levene
leven_stat, leven_p = levene(male, female)
print(f'(레벤)통계량 : {leven_stat:.4f}, p-value : {leven_p:.4f}')
if leven_p > 0.05:
    print('분산이 같다고 할 수 있다.(등분산성을 가정한다.)')
else:
    print('분산이 같다고 할 수 없다.(등분산성 가정이 부적절하다.)')
# 만약 등분산성 가정이 부적절한 경우, Welch's t-test 사용을 권장
welch_result = stats.ttest_ind(male,female, equal_var = False)   
# 위에서 말햇잖아 등분산성 가정이 부적절한 경우 . . .그래서 eql_var = False
print(welch_result)
# TtestResult(statistic=np.float64(1.233193127514512), 
# pvalue=np.float64(0.2595335362303284), 
# df=np.float64(6.613033864755501))


# 실습)두 가지 교육 방법에 따른 평균 시험 점수에 대한 검정 수행 two_sample.csv