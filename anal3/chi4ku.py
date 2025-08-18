# 이원카이제곱검정 - 교차분할표를 사용
# 변인 이 두 개 - 독립성 또는 동질성 검사
# 독립성(관련성) 검정
# - 동일 집단의 두 변인(학력수준과 대학진학 여부)을 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.

# 검정실습 1: 교육 방법에 따른 교육생들의 만족도 분석 동질성 검정 survey_method csv

# 대립 가설(H0): 교육 방법과 교육생의 만족도는 관련이 있다.
# 귀무 가설(H1): 교육 방법과 교육생의 만족도는 관련이 없다.

import pandas as pd
import scipy.stats as stats

survey_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
print(survey_data)

print(survey_data['method'].unique())
print(set(survey_data['survey']))

# 교차표 작성
ctab = pd.crosstab(index=survey_data['method'], columns=survey_data['survey'])
ctab.columns = ['매우 만족', '만족', '보통', '불만족', '매우 불만족']
ctab.index = pd.Index(['방법1', '방법2', '방법3'])
print(ctab)

chi2, p, ddof, _ = stats.chi2_contingency(ctab)
msg = "test statistics: {}, p-value: {}, df:{}"
print(msg.format(chi2, p, ddof))

# 해석
if p > 0.05:
    print("귀무 가설 채택, 교육 방법과 교육생들의 만족도는 관련이 없습니다.")
else:
    print("귀무 가설 기각, 교육 방법과 교육생들의 만족도는 관련이 있습니다.")

# 검정실습 2: 연령대별 sns 이용률의 동질성 검정
# 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로 연령대별로 홍보
# 전략을 세우고자 한다.
# 연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자.

# 대립 가설(H0): 연령대별 SNS 서비스 이용 현황은 동일하다.
# 귀무 가설(H1): 연령대별 SNS 서비스 이용 현황은 동일하지 않다.
sns_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')
print(sns_data)

print(sns_data['age'].unique())
print(sns_data['service'].unique())

# 교차표 작성
ctab = pd.crosstab(index=sns_data['age'], columns=sns_data['service'])
print(ctab)

chi2, p, ddof, _ = stats.chi2_contingency(ctab)
msg = "test statistics: {}, p-value: {}, df:{}"
print(msg.format(chi2, p, ddof))

# 해석
if p > 0.05:
    print("귀무 가설 채택, 연령대와 서비스 이용은 관련이 없습니다.")
else:
    print("귀무 가설 기각, 연령대와 서비스 이용은 관련이 있습니다.")

# 위 데이터는 샘플 데이터이다. 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하자
# 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본을 추출해 처리해보자.

print('------------------여기서부터 모집단 추정 데이터------------------')

sample_data = sns_data.sample(n=50, replace=True, random_state=1)
print(len(sample_data))

ctab3 = pd.crosstab(index=sample_data['age'], columns=sample_data['service'])
print(ctab3)
chi2, p, ddof, _ = stats.chi2_contingency(ctab3)
msg = "test statistics: {}, p-value: {}, df:{}"
print(msg.format(chi2, p, ddof))

