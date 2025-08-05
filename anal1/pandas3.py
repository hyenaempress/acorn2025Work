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
print(df1 - ser1) #브로드 캐스팅 연산이 된다. 

#연산 뚝닥 해봅시다

print('기술적 통계 관련 함수')
df = DataFrame([[1.4, np.nan],[7, -4.5], [np.nan, None], [0.5, -1]], columns=['one','two']) #결측치로 표기된다
print(df)
#결측값에 대한 이야기를 해본다 결측치 
print(df.isnull())
print(df.notnull()) # 널값을 강제할때 이런 방법이 있다 
print(df.drop(0)) # 특정 행 삭제 NaN 과 관계가 없음 
print(df.dropna()) # NaN값이 포함된 모든 행 삭제 NaN 관련된 삭제
print(df.dropna(how='any')) # 위에가 디폴트이고 결측치가 들어있으면 날린다 
#엑셀에서 필요없는 값이 들어있을 떄 날리기 좋다 
print(df.dropna(how='all')) # 행에 들어있는 모든 열값들이 NAN 이면 지움 
print(df.dropna(subset=['one'])) #서브셋으로 조건을 줄 수 있습니다 one이라는 열에 nan 이 들어있으면 그 행을 삭제합니다. 

print(df.dropna(axis='rows'))
print(df.dropna(axis='columns'))
print(df.fillna(0)) #실무에서 널을 채우고 싶을떄 대표값이나 평균값이나 최빈값 등으로 채워 줄 수있습니다. 이전값 다음값 등 다양한 것이 가능합니다. 

#기술적 통계
print('기술적 통계')
print(df.sum()) #열의 합 - NaN은 연산 X 
print(df.sum(axis=0))
print(df.sum(axis=1))
#더하기 빼기 표준편차 나누기 다 있습니다 행단위로 다 할 수 있습니다. 이 내용은 넘파이에서 했죠? 

print(df.describe())#요약통계량을 구할때 합니다. 4분위 수가 나옵니다 정말 많이 쓰입니다. 
print(df.info()) # 인포 함수는 구조를 보여줍니다 메모리 전체를 얼마나 썼는지 등 

#재구조화
print('재구조화 스택 언스택, 구간 설정, 그룹별 연산 agg 함수')
# 이 외에도 다양한 것 이 있지만 자주쓰이는것 
df = DataFrame(1000 + np.arange(6).reshape(2,3),index=['서울','대전'], columns=['2020','2021','2022'])
print(df.T) #트랜스포즈 할 수 있습니다 재구조화 
#stack, unstack
df_row = df.stack() #열 => 행으로 변경 , 열 쌓기 자주 등장 합니다.
print(df_row)

df_col = df_row.unstack() #행 -> 열로 복원 
print(df_col)

#구간 설정 한번 해보겠습니다. 

