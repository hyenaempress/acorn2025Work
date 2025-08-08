# tips.csv로 요약 처리 후 시각화
import pandas as pd
import matplotlib.pyplot as plt

tips = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv")
print(tips.info())

tips['gender'] = tips['sex']
del tips['sex']
print(tips.head(3))

# tip 비율: 파생 변수
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips.head(3))

tip_pct_group = tips['tip_pct'].groupby([tips['gender'], tips['smoker']])
print(tip_pct_group.sum())
print(tip_pct_group.max())
print(tip_pct_group.min())

result = tip_pct_group.describe()
print(result)

print(tip_pct_group.agg('sum'))
print(tip_pct_group.agg(['mean']))
print(tip_pct_group.agg('var'))

