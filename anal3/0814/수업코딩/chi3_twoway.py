# 이원카이제곱검정 - 교차분할표를 사용
# 변인 이 두 개 - 독립성 또는 동질성 검사
# 독립성(관련성) 검정
# - 동일 집단의 두 변인(학력수준과 대학진학 여부)을 대상으로 관련성이 있는가 없는가?
# - 독립성 검정은 두 변수 사이의 연관성을 검정한다.

# 실습: 교육 수준과 흡연율 간의 관련성 분석 smoke.csv

# 대립 가설: 교육 수준과 흡연율 간에 관련성이 있다.(독립이다, 연관성이 없다)
# 귀무 가설: 교육 수준과 흡연율 간에 관련성이 없다.(독립이 아니다, 연관성이 있다)

import pandas as pd
import scipy.stats as stats # 모듈을 사용한 이유: 가장 일반적으로 사용하는 라이브러리

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/smoke.csv")
print(data.head(2))

print(data['education'].unique()) # [1 2 3]
print(data['smoking'].unique()) # [1 2 3]

# 학력 수준 별 흡연 인원 수 교차 분할표
ctab = pd.crosstab(index=data['education'], columns=data['smoking'])
# ctab = pd.crosstab(index=data['education'], columns=data['smoking'], normalized = True)

ctab.index = pd.Index(["대학원졸", "대졸", "고졸"])
ctab.columns = ["과흡연", "보통", "노담"]

chi2, p, dof, expected = stats.chi2_contingency(ctab)
msg = "test statics: {}, p-value: {}, dof: {}, expected: {}" # test statics: 18.9, p-value: 0.0008, dof: 4, expected: [[68.9 83.8  58.2]
print(msg.format(chi2, p, dof, expected))

# 결론 : p-value 값이 유의수준(0.05)보다 작으므로 귀무가설을 기각
# 따라서 교육 수준과 흡연율 간에 관련이 있다.
# .....

print("음료 종류와 성별 간의 선호도 차이 분석") # 이원 카이 제곱 검정
# 남성과 여성의 음료 선호는 서로 관련이 있다 / 없다

# 대립 가설(H1): 성별과 음료 선호는 서로 관련이 있다.(성별에 따라 선호가 같다)
# 귀무 가설(H0): 성별과 음료 선호는 서로 관련이 없다.(성별에 따라 선호가 다름)
data = pd.DataFrame({
    '게토레이': [30, 20],
    '포카리스웨트': [20, 30],
    '비타500': [10, 30],
}, index=['남성', '여성'])
print(data)

chi2, p, dof, expected = stats.chi2_contingency(data)
# test statics: 11.3, p-value: 0.003, dof: 2, expected: [[21.4 21.4 17.1]
msg = "test statics: {}, p-value: {}, dof: {}, expected: {}"
print(msg.format(chi2, p, dof, expected))

# 결론 : p-value 값이 유의수준(0.05)보다 작으므로 귀무가설을 기각(귀무기각)
# 따라서  성별에 따라 음료 선호는 서로 관련이 있다.


# 시각화: heatmap
# 히트맵은 색상을 활용해 값은 분포를 보여주는 그래프
# 히스토그램이 하나의 변수에 대한 강도(높이)를 활용할 수 있다면,
# 컬러맵은 색상을 활용해 두개의 기준(x축 + y축)에 따른 강도(색상)을 보여준다고 생각하면 된다.
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
sns.heatmap(data, annot=True, fmt='d', cmap='Blues')
plt.title('성별에 따른 음료 선호')
plt.xlabel('음료 종류')
plt.ylabel('성별')
plt.show()