# Pandas 핵심 연산 및 데이터 처리 가이드

## 1. Series와 DataFrame 연산 (브로드캐스팅)

### Series 연산
```python
from pandas import Series, DataFrame
import numpy as np

s1 = Series([1,2,3], index=['a','b','c'])
s2 = Series([1,2,3,4], index=['a','b','d','c'])

# 기본 연산 (자동 브로드캐스팅)
print(s1 + s2)  # 인덱스가 일치하지 않으면 NaN 반환

# 명시적 연산 메서드
print(s1.add(s2))      # 덧셈
print(s1.multiply(s2)) # 곱셈 (sub, div도 동일하게 사용)
```

### DataFrame 연산
```python
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), 
                index=['서울', '대전', '대구'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), 
                index=['서울', '대전', '제주', '수원'])

# 기본 연산 - 일치하지 않는 인덱스는 NaN
print(df1 + df2)

# fill_value로 결측치 처리
print(df1.add(df2, fill_value=0))  # NaN을 0으로 채운 후 연산

# DataFrame과 Series 간 브로드캐스팅
ser1 = df1.iloc[0]
print(df1 - ser1)  # 각 행에서 첫 번째 행을 빼는 연산
```

## 2. 결측치(NaN) 처리

### 결측치 확인
```python
df = DataFrame([[1.4, np.nan], [7, -4.5], [np.nan, None], [0.5, -1]], 
               columns=['one','two'])

print(df.isnull())   # 결측치 확인 (True/False)
print(df.notnull())  # 비결측치 확인
```

### 결측치 제거
```python
# 행 단위 제거
print(df.dropna())                    # NaN이 포함된 모든 행 삭제
print(df.dropna(how='any'))          # 기본값: 하나라도 NaN이면 삭제
print(df.dropna(how='all'))          # 모든 값이 NaN인 행만 삭제
print(df.dropna(subset=['one']))     # 특정 열의 NaN만 고려하여 삭제

# 열 단위 제거
print(df.dropna(axis='columns'))     # NaN이 포함된 열 삭제
```

### 결측치 채우기
```python
print(df.fillna(0))  # 0으로 채우기
# 실무에서는 평균값, 최빈값, 이전값, 다음값 등으로 채움
```

## 3. 기술적 통계

### 기본 통계 함수
```python
print(df.sum())        # 열별 합계 (NaN 제외)
print(df.sum(axis=0))  # 열별 합계 (기본값)
print(df.sum(axis=1))  # 행별 합계

print(df.describe())   # 요약통계량 (평균, 표준편차, 4분위수 등)
print(df.info())       # 데이터프레임 구조 정보
```

## 4. 데이터 재구조화

### 전치(Transpose)와 Stack/Unstack
```python
df = DataFrame(1000 + np.arange(6).reshape(2,3),
               index=['서울','대전'], 
               columns=['2020','2021','2022'])

print(df.T)           # 전치 (행↔열 바꾸기)
df_row = df.stack()   # 열 → 행으로 변환 (넓은 형태 → 긴 형태)
df_col = df_row.unstack()  # 행 → 열로 복원
```

## 5. 구간 설정 (범주화)

### 수동 구간 설정
```python
import pandas as pd

price = [10.3, 5.5, 7.8, 3.6]
cut = [3, 7, 9, 11]  # 구간 경계값

result_cut = pd.cut(price, cut)
print(result_cut)  # (3, 7], (3, 7], (7, 9], (3, 7]
print(pd.value_counts(result_cut))  # 각 구간별 빈도
```

### 자동 구간 설정 (분위수 기준)
```python
datas = pd.Series(np.arange(1, 10001))
result_cut2 = pd.qcut(datas, 3)  # 3개 구간으로 균등 분할
print(pd.value_counts(result_cut2))
```

## 6. 그룹별 연산

### GroupBy와 집계함수
```python
group_col = datas.groupby(result_cut2)

# 내장 집계함수 사용
print(group_col.agg(['count','mean','std','min']))

# 사용자 정의 함수
def myFunc(gr):
    return {
        'count': gr.count(),
        'mean': gr.mean(),
        'std': gr.std(),
        'min': gr.min()
    }

print(group_col.apply(myFunc))
```

## 실무 활용 팁

1. **결측치 처리**: 데이터 분석 전 반드시 확인하고 적절한 방법으로 처리
2. **브로드캐스팅**: DataFrame과 Series 간 연산 시 자동으로 적용됨
3. **구간 설정**: 연속형 데이터를 범주형으로 변환할 때 유용
4. **그룹별 연산**: 카테고리별 통계량 계산에 필수적
5. **재구조화**: 데이터 시각화나 분석을 위해 형태 변환이 필요할 때 사용

## 다음 단계
더 구체적인 실무 예시와 피벗 테이블, 고급 그룹별 연산 등을 다루는 심화 내용으로 이어집니다.