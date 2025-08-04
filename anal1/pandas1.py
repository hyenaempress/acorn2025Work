#pandas : 행과 열이 단순 정수형 인덱스가 아닌 레이블로 식별되는 데이터 구조 numpy의 구조화된 배열들을 보완(강)한 모듈이다.
#고수준의 자료구조 (시계열 축약연산, 누락데이터 처리, SQL, 시각화) (Series, DataFrame)와 데이터 분석 도구를 제공
import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
#pandas는 numpy를 기반으로 하며, numpy의 배열을 pandas의 데이터 구조로 변환할 수 있다.

# Series : 일련의 객체를 담을 수 있는  1차원 배열과 유사한 구조로, 인덱스와 값으로 구성된 데이터 구조 인덱스 색인을 가진
#list, array, dict 등 다양한 형태의 데이터를 담을 수 있다.

obj = Series([3, 7, -5, 4])
print(obj)

#인덱싱이 자동으로 따라 붙는다. 

obj = Series((3, 7, -5, 4))
print(obj, type(obj)) 
#타입을 보면 시리즈라는 것을 알 수 있다
#자동 인덱싱 라벨 명시적이다. 넘파이는 묵시적이다. 
#리스트랑 튜플은 순서가 있어 가능하지만 Set은 순서가 없기 때문에 불가능하다.
print(obj[1])       

obj2 = Series([3, 7, -5, 4], index=['a', 'b', 'c', 'd'])
print(obj2)
#인덱스는 문자열로 지정할 수 있다.
print(obj2.sum(), np.sum(obj2))  # 합계
#급할때는 파이썬의 sum() 함수를 사용해도 된다.

print(obj2.values) # 값들만 출력
print(obj2.index) # 인덱스들만 출력


print(obj2[1:3])  # 슬라이싱
print(obj2['a'])  # 인덱스로 값 접근 값만 나와요 
print(obj2[['a']])  # 인덱스로 값 접근 (Series 형태로 반환) 차원 형태로 분할 
#print(obj2['a','d'])  # 인덱스 두개 불러오기
print(obj2[['a', 'd']])  # 인덱스 두개 불러오기
print(obj2['a':'d'])  # 인덱스 슬라이싱

#위의 방법으로 운영이 가능하다.
print(obj2[3])  #
print(obj2.iloc[3])  # 인덱스 위치로 값 접근
print(obj2.iloc[[2,3]])     # 인덱스 위치로 값 접근 (여러 개)
#iloc 를 운영 할 수 있다. 

print(obj2 > 0)  # 불리언 인덱싱 (0보다 큰 값들
print(obj2[obj2 > 0])  # 불리언 인덱싱 (0보다 큰 값들만 선택)
print('a' in obj2)  # 인덱스 존재 여부 확인
print('aa' in obj2)  # 인덱스 존재 여부 확인

#시리즈를 만들때 리스트나 튜플을 만든다 
#딕셔너리로 시리즈 만들기
print("\n=== 딕셔너리로 시리즈 만들기 ===")
data = {'a': 3, 'b': 7, 'c': -5, 'd': 4}


#딕셔너리로 시리즈를 만들면 인덱스가 딕셔너리의 키가 된다.
#딕셔너리로 시리즈 만들기 (인덱스 지정)
names = { 'mouse' :5000, 'keyboard': 10000, 'monitor': 20000 }
#파이썬 초기 버전엔 딕셔너리도 순서가 없었는데 생겼다 ,
print(names, type(names))  # 딕셔너리

obj3 = Series(names)

print(obj3)
print(obj3)  # 딕셔너리로 시리즈 만들기
#키가 인덱스가 되고, 값이 시리즈의 값이 된다.
obj3.index = ['마우스', '키보드', '모니터']
print(obj3)  # 인덱스 변경 후 시리즈 출력
print(obj3['마우스'])  # 인덱스로 값 접근
#print(obj3[0])  # 인덱스 위치로 값 접근 

obj3.name = '전자제품 가격'
print(obj3)  # 시리즈 이름 추가 후 출력


print("\n=== 데이터 프레임 ===")
# 데이터 프레임: 2차원 테이블 형태의 데이터 구조
# 행과 열로 구성된 데이터 구조로, 각 열은 시리즈 형태로 저장된다.
# 데이터 프레임은 행과 열이 있는 2차원 배열로 생각할
#DataFrame은 Series 객체가 모여 표를 구성하는 것이다. 

df = DataFrame(obj3)
print(df)  # 데이터 프레임 출력

#시리즈를 가지고 데이터 프레임을 만들었다. 

data = {
    'irum': ['홍길동' , '한국인' , '신기해', '공기밥' ,'한가해'],
    'juso': ('역삼동','신당동', '역삼동', '역삼동', '신사동'),
    'nai': [23, 25, 33, 30, 35],
}

frame = DataFrame(data)
print(frame)  # 데이터 프레임 출력

#딕트를 이용해서 데이터 프레임을 만들었습니다. 표의 묘양을 취하는 
#멤버에 접근해서 컬럼을 얻어내고 싶을 떄는 ? 

print(frame['irum'])  # 'irum' 열 선택
print(frame.irum) #이걸 타입으로 감싸 봅니다.
print(type(frame.irum)) #<class 'pandas.core.series.Series'>

#열 순서 바꾸기
print("\n=== 열 순서 바꾸기 ===")
print(DataFrame(data, columns=['juso', 'irum', 'nai']))  # 열 순서 변경
#열 순서를 바꿔서 데이터 프레임을 만들었습니다.

#데이터에 NaN 값 추가
print("\n=== NaN 값 추가 ===")
frame2 = DataFrame(data, columns=[ 'irum', 'nai', 'juso', 'tel'],
                   index=['a', 'b', 'c', 'd', 'e'] ) #인덱스 부여 숫자에서 문자로 변경  
print(frame2)  # NaN 값이 있는 데이터 프레임 출력

#tel에 값 부여 
#하나로 밀어넣기
frame2['tel'] = '111-1111-1111'  # 모든 행에 동일한 값 추가
print(frame2)  # 'tel' 열에 값 추가 후 데이터 프레임 출력
frame2['tel'] = ['010-1234-5678', '02-9876-5432', np.nan, '031-1111-2222', '02-3333-4444']
print(frame2)  # 'tel' 열에 값 추가 후 데이터 프레임 출력
val = Series(['010-1234-5678', '02-9876-5432', np.nan], index=['a', 'b', 'c'])
frame2.tel = val
print(frame2)  # 'tel' 열에 값 추가 후 데이터 프레임 출력

print(frame2.T) # 데이터 프레임 전치 (행과 열을 바꿈)
#나중에 사용할 일이 많다 

print(frame2.values,type(frame2.values))  # 데이터 프레임의 값들만 출력 다시 2차원 배열
#print(frame2.value[0,1])  # 데이터 프레임의 인덱스들만 출력
#print(frame2.value[0:2])        # 데이터 프레임의 값들 슬라이싱

#행 / 열 삭제 
frame3 = frame2.drop('d')  # 'd' 행 삭제 인덱스가 d인 행 삭제 
print(frame3)  # 행 삭제 후 데이터 프레임 출력
frame4= frame2.drop('tel', axis=1)  # 열이름이 'tel'인 열 삭제
print(frame4) # 열 삭제 후 데이터 프레임 출력

#정렬 
print("\n=== 데이터 프레임 정렬 Sort ===")
print(frame2.sort_index(axis=0, ascending=False))  # 행 인덱스 기준으로 내림차순 정렬 
#sort 알고리즘을 몰라도 데이터 분석이 가능하다. 
#하지만 알고리즘을 알아야함 
print(frame2.sort_index(axis=1, ascending=False))   # 열 인덱스 기준으로 내림차순 정렬
print(frame2.sort_index(axis=1, ascending=True))   # 열 인덱스 기준으로 내림차순 정렬
print(frame['juso'].value_counts)  # 'juso' 열 선택 이거 모르면 for문 돌려서 누적해줘야 함 

print(frame2.sort_values(by='nai', ascending=True))  # 'nai' 열을 기준으로 오름차순 정렬
print(frame2.sort_values(by='nai', ascending=False))  # 'nai' 열을 기준으로 내림차순 정렬


#문자열 자르기 
print('=== 문자열 자르기 ===')
data = {
    'juso': ['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon': [23, 25, 15],
}
fr= pd.DataFrame(data)
print(fr)  # 데이터 프레임 출력
result1 = Series([x.split()[0] for x in fr.juso])  # 'juso' 열의 문자열을 공백으로 분리하여 첫 번째 부분만 추출
result2 = Series([x.split()[1] for x in fr.juso])  # 'juso' 열의 문자열을 공백으로 분리하여 두 번째 부분만 추출
#시리즈는 리스트 타입으로 잘라 넣었더니 시리즈 타입으로 바꾸어 주었고 012숫자 인덱스를 가지고있다.
print(result1)  # 첫 번째 부분 출력
print(result2)  # 두 번째 부분 출력
print(result1.value_counts())  # 첫 번째 부분의 값 개수 세기
print(result2.value_counts())  # 두 번째 부분의 값 개수 세기
#이런식으로 인덱스를 이용해서 데이터를 다룰 수 있다.

#빅데이터로 가면 중요하게 된다.