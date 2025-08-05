from pandas import Series, DataFrame
import numpy as np 


s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([1,2,3,4], index=['a','b','d','c'])
print(s1)
print(s2)

print(s1 + s2) #자동으로 브로드 캐스팅? 

print(s1.add(s2)) #numpy 의 add 기능 

print(s1.multiply(s2)) #sub div 
print()
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['서울', '대전', '대구'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['서울', '대전', '제주', '수원'])
print(df1)
print(df2)
print(df1 + df2) #NaN 이 그대로 노출된다 
print(df1.add(df2, fill_value=0)) #NaN은 0으로 채운 후 연산에 참여 

#곱하기 
#나누기 나중에 보기 

ser1 = df1.iloc[0]
print(ser1)


