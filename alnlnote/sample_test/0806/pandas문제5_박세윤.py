# https://cafe.daum.net/flowlife/SBU0/10
#* Pandas 문제 * : 5~6번 

import pandas as pd

print('---- pandas 문제 5) -----')
print('---- pandas 문제 5-1) -----')
# 5-1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자 수를 계산한다.
# cut() 함수 : 연속형 숫자 데이터를 구간(bins)별로 나눠서 그룹화해줌

# 절대경로 읽어옴
tt = pd.read_csv(r'C:\Users\SeYun\anaconda3\envs\day0731\pandas\titanic_data.csv')

bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
# AgeGroup 컬럼 생성 + 값 넣기
# right=False : [) 좌 이상 우 미만 (예: 1~20살 → 소년)
tt['AgeGroup'] = pd.cut(tt['Age'], bins=bins, labels=labels, right=False)
# 생존자(Survived==1)만 선택
# value_counts()는 인원수를 자동으로 세줌
# reindex(labels) : 출력 순서를 labels 순으로 맞춤
a = tt[tt['Survived']==1]['AgeGroup'].value_counts().reindex(labels)
print(a)
print()

print('---- pandas 문제 5-2-1) -----')
# 5-2-1) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다. 
# df.pivot_table()사용 -> index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
# index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.

# pivot_table() : 행/열 기준으로 데이터를 요약
# values='Survived' → 생존 여부(0/1)를 계산 대상으로 사용
# aggfunc='mean' → 각 그룹별 Survived 평균 = 생존율
a = tt.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
print(a)
print()

print('---- pandas 문제 5-2-2) -----')
# 5-2-2) 위 결과물에 Age를 추가. 백분율로 표시. 
# 소수 둘째자리까지.    예: 92.86 

# observed=False 모든 가능한 카테고리 조합을 보여줌, Deprecated 경고로 인해 추가 
# *100 → 백분율로 변환
a = tt.pivot_table(index=['Sex','AgeGroup'], columns='Pclass', values='Survived'
                   , aggfunc='mean',observed=False)*100  
# round(2) → 소수 둘째자리까지 반올림
a = a.round(2)
print(a) 
print()

print('---- pandas 문제 6) -----')
print('---- pandas 문제 6-1) -----')
#  6-1) human.csv 파일을 읽어 아래와 같이 처리하시오.
#      - Group이 NA인 행은 삭제
#      - Career, Score 칼럼을 추출하여 데이터프레임을 작성
#      - Career, Score 칼럼의 평균계산
#      참고 : strip() 함수를 사용하면 주어진 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제시켜 준다. 
#              그래서 위의 문자열에서 \n과 오른쪽에 있는 공백이 모두 사라진 것을 확인할 수 있다. 
#              주의할 점은 strip() 함수는 문자열의 양 끝에 있는 공백과 \n을 제거해주는 것이지 중간에 
#              있는 것까지 제거해주지 않는다.

h = pd.read_csv(r'C:\Users\SeYun\anaconda3\envs\day0731\pandas\human.csv')

# .str.strip() → 문자열 양쪽 공백 제거
h.columns = h.columns.str.strip()
# DataFrame에서 object 타입(문자열) 컬럼만 선택
str_cols = h.select_dtypes(include='object').columns
# apply → DataFrame의 각 컬럼 하나를 col에 전달, 변환식 람다
# col.str.strip() 실행 → 컬럼의 모든 문자열에서 양쪽 공백 제거
h[str_cols] = h[str_cols].apply(lambda col: col.str.strip())
# Group 'NA'가 아닌 행만 남김
h = h[h['Group'] != 'NA']
# 평균계산
print(h[['Career','Score']].mean())
print()

print('---- pandas 문제 6-2) -----')
# 6-2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
#      - 파일 정보 확인
#      - 앞에서 3개의 행만 출력
#      - 요약 통계량 보기
#      - 흡연자, 비흡연자 수를 계산  : value_counts()
#      - 요일을 가진 칼럼의 유일한 값 출력  : unique()
#           결과 : ['Sun' 'Sat' 'Thur' 'Fri']

t = pd.read_csv(r'C:\Users\SeYun\anaconda3\envs\day0731\pandas\tips.csv')

# 파일 정보
print(t.info())
print()
# 앞 3행 출력
print(t.head(3))
print()
# 요약 통계량
print(t.describe())
print()
# smoker Y/N 수
print(t['smoker'].value_counts())
print()
# 중복X , day 유일한 값
print(t['day'].unique())