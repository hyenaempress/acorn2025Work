# 교차 분석(카이제곱) 가설검정, cross-tabulation analysis
# 데이터를 파악할 때 중심위치(평균)와 퍼짐 정도(분산)가 중요한 데
# 카이(chi)제곱은 바로 분산에 대한 분포다.
# 범주형 자료를 대상으로 교차 빈도에 대한 기술 통계량 내지 검정 통계량, 유의성을 검증해주는 추론통계 기법(표본 데이터를 갖고)
# 일원 카이 제곱 검정(변인: 단수) - 적합도, 선호도 검사 | 교차 분할 표 미 사용
# 이원 카이 제곱 검정(변인: 복수) - 독립성, 동질성 검사 | 교차 분할 표 사용 -> 더 많이 사용해요
# 유의확률: p-value에 의해 집단 간의 차이 여부를 가설로 검증
# 카이 제곱 검증 정식 명칭: Pearson's Chi-squared test <-- 두 불연속변수(범주형) 간의 상관관계를 측정하는 기법

# 교차분석 흐름 이해용 : 
# (1)수식에 의해 chi^2 값 구하기 - (관측값 - 기대값)^2 / 기대값의 전체합에 의해 chi^2 값 구하기
# (2)함수로 구하기 - scipy.stats.chi2_contingency() 사용

import pandas as pd
data = pd.read_csv("./01_big_data_machine_learning/data/pass_cross.csv")

# 대립가설(H0): 벼락치기 공부하는 것과 합격 여부는 관계가 없다.
# 귀무가설(H1): 벼락치기 공부하는 것과 합격 여부는 관계가 있다.

print(data[(data['공부함'] == 1) & (data['합격'] == 1)].shape[0]) # 18명
print(data[(data['공부함'] == 1) & (data['불합격'] == 1)].shape[0]) # 7명

# 빈도표
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
ctab.columns = ['합격', '불합격', '행합']
ctab.index = pd.Index(['공부함', '공부안함', '열합'])
print(ctab)

# 방법1: 수식 사용
# 기대 값: (각 행의 주변 합) * (각 열의 주변 합) / 전체 합
# 카이 제곱 통계량: (관측값 - 기대값)^2 / 기대값

# 임계값은? -> 카이제곱 표 사용

# 자유도(df) = (행의 수 - 1) * (열의 수 - 1)

# cv(critical value) = 3.84

# 결론: 카이^2의 검정 통계량 3.0은 cv(3.84) > 3.0 이므로 귀무가설을 채택한다.

# 방법2: 함수 사용 - p-value 판정
import scipy.stats as stats

chi2, p, dof, expected = stats.chi2_contingency(ctab)
print(chi2, p, dof, expected)
# chi2 : 3.0, p:  0.5578
msg = "Test statistic: {:.2f}, p-value: {:.4f}".format(chi2, p)
print(msg.format(chi2, p))
# 결론: p-value(0.5578) > 유의수준(0.05) 이므로 귀무가설을 채택한다.

# 새로운 주장을 위해 수집된 data는 (필연이 아니라) 우연히 발생한 데이터