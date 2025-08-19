# 독립 표본 검증 : 두 집단의 평균의 차이 검정
# 서로 다른 두 집단의 평균에 대한 통계 검정에 주로 사용된다.
# 비교를 위해 평균과 표준편차 통계량을 사용한다.
# 두 집단의 평균값의 차이가 얼마인지, 표준편차는 얼마나 다른지 확인하여
# 분석 대상인 두 자료가 같을 가능성이 우연의 범위 5%에 들어가는지를 판별
# 결국, t-test는 두 집단의 평균과 표준편차 비율에 대한 대조 검정법이다.

# * 서로 독립인 두 집단의 평균 차이 검정(independent samples t-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(two sample)이라고 한다.

# 실습 1) 남녀 두 집단 간 파이썬 시험의 평균 차이 검정
# 95%의 신뢰수준에서 우연히 발생할 확률이 5% 보다 작냐?
# 선행 조건: 두 집단 자료는 정규 분포를 따른다. 등분산성(분산이 동일하다)
from scipy import stats
import pandas as pd
import numpy as np

male = [75, 85, 100, 72.5, 86.5]
female = [63.2, 76, 52, 100, 70]
print(np.mean(male), ' ', np.mean(female)) # 83.8   72.24
print(np.std(male), ' ', np.std(female))

# two_sample = stats.ttest_ind(male, female)
two_sample = stats.ttest_ind(male, female, equal_var = True) # statistic=1.233, pvalue=0.252, df=8.0
print(two_sample) # statistic=1.233, pvalue=0.252, df=8.0
# 해석
if two_sample[1] > 0.05:
    print("두 집단 간 평균 차이는 통계적으로 유의미하지 않다.")
else:
    print("두 집단 간 평균 차이는 통계적으로 유의미하다.")


# 실습 2) 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample.csv'
# 귀무: 
# 대립: 

print("\n등분산 검정")
from scipy.stats import levene
leven_stat, leven_p = levene(male, female)
print(f"통계량:{leven_stat:.4f}, p-value:{leven_p:.4f}") # 0.5095, p-value:0.4957
if leven_p > 0.05:
    print("분산이 같다고 할 수 있다.")
else:
    print("분산이 같다고 할 수 없다. 등분산 가정이 부적절")

# 만약 등분산성 가정이 부적절한 경우 Welch's t-test 사용을 권장
welch_result = stats.ttest_ind(male, female, equal_var=False)
print(welch_result) # pvalue = 0.259