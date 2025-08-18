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
# 검정실습 2: 연령대별 sns 이용률의 동질성 검정
# 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로 연령대별로 홍보
# 전략을 세우고자 한다.
# 연령대별로 이용 현황이 서로 동일한지 검정해 보도록 하자.

# 대립 가설(H0): 연령대별 SNS 서비스 이용 현황은 동일하다.
# 귀무 가설(H1): 연령대별 SNS 서비스 이용 현황은 동일하지 않다.
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
print(data.head(2))
print(data['method'].unique()) # [1 2 3]
print(set(data['survey'])) # {1, 2, 3, 4, 5}

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족', '만족', '보통', '불만족', '매우불만족']
ctab.index = ['방법1', '방법2', '방법3']
print('ctab:\n', ctab)

chi2, p, dof, _ = stats.chi2_contingency(ctab)
msg = 'test statistic:{}, p-value:{}, dof:{}'
print(msg.format(chi2, p, dof)) # test statistic:6.544667820529891, p-value:0.5864574374550608, dof:8
# 결론 : p-value(0.586) > 유의수준(0.05)이므로 귀무가설 채택

#수업 녹음 2 
data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')
print(data2.head(3))

print(data2['age'].unique()) 
print(data2['service'].unique())

ctab2 = pd.crosstab(index=data2['age'], columns=data2['service']) # margins=True  
#교차표는 나왔고 결과를 받아보년 됩니다.

print('ctab2:\n', ctab2)

chi2, p, dof, _ = stats.chi2_contingency(ctab2)
msg = 'test statistic:{}, p-value:{}, dof:{}'
print(msg.format(chi2, p, dof)) # test statistic:10.450867051968448, p-value:0.005395017319603369, dof:8
#귀무 기각이 됩니다. 대립가설 채택

#사실 위 데이터는 셈플데이터이다. 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하자. 
# 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본을 추출해 보자 
sample_data = data2.sample(n=50, replace=True, random_state=1)
print(sample_data.head(3))

ctab3 = pd.crosstab(index=sample_data['age'], columns=sample_data['service'])
print('ctab3:\n', ctab3)
chi2, p, dof, _ = stats.chi2_contingency(ctab3)
msg = 'test statistic:{}, p-value:{}, dof:{}'
print(msg.format(chi2, p, dof)) # test statistic:10.450867051968448, p-value:0.005395017319603369, dof:8
#귀무 기각이 됩니다. 대립가설 채택

    


