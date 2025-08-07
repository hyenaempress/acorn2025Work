import pandas as pd

url = 'https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv'
df = pd.read_csv(url) 

print('\n\n 1) : 파일정보확인')
df.info() # 프린트하지않아도 타입이랑, 인덱스범위, 칼럼몇개인지 , 데이터타입 , 메모리사용량 알려준다
print('\n\n 2) : 앞에서 3개의 행만 출력')
print(df.head(3)) #3개행 출력
print('\n\n 3) : 요약 통계량 보기')
print(df.describe()) #요약통계량 보기
print('\n\n 4) : 흡연자,비흡연자수를 계산 : value_counts()')
print(df.smoker.value_counts()) #흡연자, 비흡연자수 계산 value_counts()사용
print('\n\n 5) : 요일을 가진 칼럼의 유일한 값 출력 : unique()')
print(df.day.unique())#요일을 가진 칼럼의 유일한 값 출력 unique()
print('\n\n')







