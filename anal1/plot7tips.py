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

#사용자 정의 함수 한번 만들게요 
def myFunc(group):
    diff= group.max() - group.min()
    return diff


#내장된 함수일땐 그냥 쓰면 되고 사용자 정의 함수일땐 그냥 쓰면 안되고 함수 이름을 써줘야 한다.  (이건 추가된 내용이빈다.)
result2 = tip_pct_group.agg(['sum', 'mean', 'var','max', myFunc])
print(result2) 

result2.plot(kind='barh', title='agg fumc result',stacked=True)
plt.show()

#두루두루 쭉 해봤어용 ㅎㅎ.,, 넘파이 판다스 지금 쭉쭉쭉 잘 지나오면서 어떻게 쓰는지 잘 봤고 시각화도 그때그때 필요할때마다 할거에요
#독립적으로 운영할때는 파일로 저장하지만 웹에서 시각화 할때는 이미지를 만들어서 이미지를 만들땐 chart.js 라는 라이브러리를 써요 
#저번에도 했던 이야기이기도 하지만 중요하니까 다시 언급합니다. 
#시각화 이야기는 이정도로 마무리 하겠습니다! 
#이제부터는 데이터 베이스에 대한 이야기를 할게요 






