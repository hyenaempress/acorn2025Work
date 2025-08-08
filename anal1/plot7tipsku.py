#tips.csv로 요약 처리 후 시각화 

import pandas as pd 
import matplotlib.pyplot as plt


tips = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv')

print(tips.info())

#컬럼 명 한번 바꿔볼게요. 넘겨주고 받아주고  
tips['gender'] = tips['sex'] #컬럼 하나 또 생긴다 
del tips['sex'] 
print(tips.head(3))

#비율을 한번 구하겠습니다. Tib을 토탈 비율로 나눠서 구하겠습니다. 

#파생 변수 한번 만들어 봅니다.
tips['tip_pct'] = tips['tip'] / tips['total_bill'] 
print(tips.head(3)) #파생 변수하나 만든다 

#젠더별 스모커에 대한 그룹 바이를 해볼까?
tip_pct_group = tips.groupby([tips['gender'], tips['smoker']]).mean()
print(tip_pct_group)

print(tip_pct_group.sum()) #비율 합계 
print(tip_pct_group.max()) #비율 합계 
print(tip_pct_group.min()) #비율 합계 

result = tip_pct_group.describe()
print(result)

#tip_pct_group.sum(axis=1)
print(tip_pct_group.agg('sum')) #비율 합계 
print(tip_pct_group.agg(['sum', 'mean']))
print(tip_pct_group.agg('var')) #비율 합계 

# 우리 뭐한번 할까유... 함수를 실행시켜서 그래프를 그려볼려고 해요

result2 = tip_pct_group.agg(['sum', 'mean', 'var','max'])
print(result2) 


