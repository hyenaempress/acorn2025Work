#병합에 대한 이야기를 하겠음, 데이터 프레임 병합 
from pandas import Series, DataFrame
import pandas as pd 
import numpy as np


df1 = pd.DataFrame({'data1':range(7), 'key': ['b','b','b','c','a','a','b']})
print(df1)
df2 = pd.DataFrame({'key': ['a','b','d'] })
print(df2)
print()
print(pd.merge(df1, df2)) #기준치가 없이 머지할 수 없습니다 
#바로 키가 기준치가 됩니다. 
print(pd.merge(df1, df2, on = 'key')) #키열을 기준으로 병합을 한 것입니다. 
#양쪽에 있는 것만 참여합니다 (이너 조인 형 )
print(pd.merge(df1, df2, on='key', how='inner'))
print()
print(pd.merge(df1, df2, on='key', how='outer')) #아우터 조인입니다 없는 것은 nan 처리 합니다. 없는 애는 결측치가 됩니다. 
print()
print(pd.merge(df1, df2, on='key', how='left')) #left outher
print()
print(pd.merge(df1, df2, on='key', how='right')) #right outher


print('공통칼럼이 없는 경우')
df3 = pd.DataFrame({'key': ['a','b','c'], 'data2':range(3)})
print(df3)
#df1 하고 따졌을때 공통 칼럼이 없음 
print(df1)
#지정해주면 됨 
print(pd.merge(df1, df3, left_on='key' , right_on= 'key2')) #머지를 할 수 있는 요건이 조성됨 
#inner 조인을 하게되고 머지를 할 수 있음 

print('---------')
print(pd.concat([df1, df3], axis=1)) #열단위로 갈거면 0이다 이런식으로 컨켓을 통해서 할 수 있다 

s1 = pd.Series([0.1], index=['a', 'b'])
s2 = pd.Series([2,3,4], index=['c','d','e'])
s3 = pd.Series([5,6], index= ['f','g'])
print(pd.concat([s1,s2,s3], axis=0))

print('그룹화 : pivot_table')
#피봇 테이블에 대한 이야기를 하겠습니다.
#그룹화 연산을 할 수 있습니다. 행과 열만 움직여 가면서 재구성 합니다

data = {'city': ['강남','강북','강남','강북'],
        'year': [2000, 2001, 2002, 2003],
        'pop': [3.3, 2.5, 3.0, 2.0]
        }

df = pd.DataFrame(data)
print(df)
print(df.pivot(index='city', columns='year', values='pop')) #
print(df.set_index(['city','year']).unstack()) #피봇으로 만든것과 같다 데터 프레임에 얘를 써줄 수 있다.
print(df.describe()) #많이 쓰이는 합수이다 요약 통계량을 보여준다. 
print('pivot_table : pivot 과 group by의 중간적 성격을 띈다')
print(df.pivot_table)
#연산을 좀 할것 
print(df)
print(df.pivot_table(index=['city']))
print(df.pivot_table(index=['city'], aggfunc = 'mean'))
print(df.pivot_table(index=['city','year'], aggfunc = [len,'sum'])) #함수의 이름을 문자열로 적으면 됩니다 파이썬의 기본 명령어는 그냥 쓴다
print(df.pivot_table(index='city', values='pop', aggfunc=',mean')) #함수의 이름을 문자열로 적으면 됩니다 파이썬의 기본 명령어는 그냥 쓴다
print(df.pivot_table(index=['city'], values=['pop'], columns=['city']))
print(df.pivot_table( values=['pop'], index=['city'], columns=['city'], margins= True))
print(df.pivot_table( values=['pop'], index=['city'], columns=['city'], margins= True, fill_value=0))
#옵션이 많습니다. 

print()
hap = df.groupby(['city']) #소계구하기
print(hap)
print(hap.sum())
#그룹바이라는 

print(df.groupby(['city']).sum())
print(df.groupby(['city','year']).mean()) 
#데이터 프레임이 하는 일은 정말 많아요! 기본은 알고갑시다. 
