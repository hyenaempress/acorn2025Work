# 카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
#   예제파일 : cleanDescriptive.csv
#   칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
#   조건 : level, pass에 대해 NA가 있는 행은 제외한다.

# 대립가설(H0): 부모의 학력 수준과 자녀의 대학 진학여부는 관련이 없다.
# 귀무가설(H1): 부모의 학력 수준과 자녀의 대학 진학여부는 관련이 있다.

import pandas as pd
import scipy.stats as stats

# 데이터 로딩
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv')
print(data)
# print(data)
# print(data['level']) # 부모의 학력 수준
# print(data['pass']) # 자녀의 대학 진학 여부

# 빈도표 구하기(부모의 학력에 따른 대학 진학 여부 CROSS TABLE)
ctab = pd.crosstab(index = data['level'], columns = data['pass'], dropna=True)
# print(ctab)
print(ctab)

# 함수 사용 - p-value 판정
chi2, p, dof, expected = stats.chi2_contingency(ctab)
print(chi2, p, dof, expected)
# Test statistic: 2.77, p-value: 0.2507
msg = "Test statistic: {:.2f}, p-value: {:.4f}".format(chi2, p)
print(msg.format(chi2, p))
# 결론: p-value(0.2507) < 유의수준(0.05) 이므로 귀무가설을 기각한다.
# 따라서, 부모의 학력 수준과 자녀의 대학 진학 여부는 관련이 있다.