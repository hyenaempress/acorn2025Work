#판다스의 여러가지 기능들 

#재색인
from pandas import Series, DataFrame
import numpy as np

# Series 재색인
data = Series([1, 3, 2], index=['1', '4', '2'])
print(data)

data2= data.reindex(['1', '2', '3', '4'])
print(data2)  # '3'은 NaN으로 채워짐

#재색인 할 때 값 채우기
data3 = data2.reindex([0,1,2,3,4,5])
print(data3)  # NaN으로 채워짐

#대응값이 없는 (NaN)인덱스는 결측값인데 777로 채우기 
data3 = data2.reindex([0,1,2,3,4,5], fill_value=777)
print(data3)  # NaN 대신 777로 채워짐

data3 = data2.reindex([0,1,2,3,4,5], method='ffill')
print(data3)  # NaN 대신 이전 값으로 채워짐

data3 = data2.reindex([0,1,2,3,4,5], method='pda')
print(data3)  # NaN 대신 다음 값으로 채워짐

data3 = data2.reindex([0,1,2,3,4,5], method='backfill')
print(data3)  # NaN 대신 다음 값으로 채워짐

#bool 처리, 슬라이싱 관련 메서드 
#복수 인덱싱 할때 사용 
#loc 는 라벨을 지원합니다, iloc 는 숫자도 지원합니다. 
df = DataFrame(np.arange(12).reshape(4, 3), index = ['1월' , '2월', '3월', '4월'], columns=['강남', '강북', '서초'])
print(df)        
print(df['강남'])  # '강남' 열 선택
print(df['강남'] > 0 )  # 불리언 인덱싱
print(df[df['강남'] > 0])  # '강남' 열이 0보다 큰 행 선택
print(df[df['강남'] > 0]['강북'])  # '강남' 열이 0보다 큰 행의 '강북' 열 선택
print(df[df['강남'] > 0][['강북', '서초']])  # '강남' 열이 0보다 큰 행의 '강북', '서초' 열 선택
print(df[df['강남'] > 0].loc['2월':'4월'])  # '강남' 열이 0보다 큰 행의 '2월'부터 '4월'까지 선택
print(df.loc[:'2월', ['서초']])
print()
print(df.iloc[2])
print(df.iloc[:3]) #인덱스가 보이는게 차이점 
print(df[df['강남'] > 0].iloc[1:3])  # '강남' 열이 0보다 큰 행의 1번째부터 2번째까지 선택
print(df[df['강남'] > 0].iloc[1:3, 1:3])  # '강남' 열이 0보다 큰 행의 1번째부터 2번째까지, '강북', '서초' 열 선택
print(df[df['강남'] > 0].iloc[1:3, [1, 2]])  # '강남' 열이 0보다 큰 행의 1번째부터 2번째까지, '강북', '서초' 열 선택
print(df[df['강남'] > 0].iloc[1:3, [0, 2]])  # '강남' 열이 0보다 큰 행의 1번째부터 2번째까지, '강남', '서초' 열 선택
print(df[df['강남'] > 0].iloc[1:3, [0, 1]])  # '강남' 열이 0보다 큰 행의 1번째부터 2번째까지, '강남', '강북' 열 선택
#넘파이는 인덱스가 묵시적입니다. 
print(df.iloc[:3, 1:3])
print(df.ilic[:3, 1:3])



