# 이원카이제곱검정 - 교차분할표를 사용
# 변인이 두 개 - 독립성 또는 동질성 검정
# 독립성(관련성) 검정
# - 동일 집단의 두 변인 학력수준과 대학진학 여부를 대상으로 관련성이 있는지 검정
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다

# 실습 1 교육방법에 다른 교육생들의 만족도 분석 동질성 검정 survey_method csv
import pandas as pd
import scipy.stats as stats

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
print(data2['con'].unique())

ctab2 = pd.crosstab(index=data2['age'], columns=data2['con'])
ctab2.index = ['20대', '30대', '40대', '50대', '60대']
ctab2.columns = ['친구', '비친구']
print('ctab2:\n', ctab2)


