# 10. 판다스 기능 세부

## 10.1 Series와 DataFrame 재색인 (Reindexing)

### Series 재색인 기본

```python
from pandas import Series, DataFrame
import numpy as np

# Series 재색인
data = Series([1, 3, 2], index=['1', '4', '2'])
print(data)

data2 = data.reindex(['1', '2', '3', '4'])
print(data2)  # '3'은 NaN으로 채워짐
```

### 재색인 시 결측값 처리

#### fill_value로 결측값 채우기
```python
data3 = data2.reindex([0, 1, 2, 3, 4, 5], fill_value=777)
print(data3)  # NaN 대신 777로 채워짐
```

#### method 옵션을 이용한 결측값 채우기
```python
# 이전 값으로 채우기 (forward fill)
data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='ffill')
print(data3)  # NaN 대신 이전 값으로 채워짐

# 다음 값으로 채우기 (backward fill)
data3 = data2.reindex([0, 1, 2, 3, 4, 5], method='backfill')
print(data3)  # NaN 대신 다음 값으로 채워짐
```

## 10.2 DataFrame 불리언 인덱싱 및 슬라이싱

### DataFrame 생성 및 기본 선택
```python
df = DataFrame(np.arange(12).reshape(4, 3), 
               index=['1월', '2월', '3월', '4월'], 
               columns=['강남', '강북', '서초'])
print(df)

# 열 선택
print(df['강남'])  # '강남' 열 선택
```

### 불리언 인덱싱
```python
# 조건문으로 불리언 Series 생성
print(df['강남'] > 0)  # 불리언 인덱싱

# 조건을 만족하는 행 선택
print(df[df['강남'] > 0])  # '강남' 열이 0보다 큰 행 선택

# 조건을 만족하는 행의 특정 열 선택
print(df[df['강남'] > 0]['강북'])  # '강남' 열이 0보다 큰 행의 '강북' 열 선택
print(df[df['강남'] > 0][['강북', '서초']])  # 여러 열 선택
```

### loc와 iloc를 이용한 고급 슬라이싱
```python
# loc: 라벨 기반 인덱싱
print(df[df['강남'] > 0].loc['2월':'4월'])  # '강남' 열이 0보다 큰 행의 '2월'부터 '4월'까지

# iloc: 정수 위치 기반 인덱싱
print(df[df['강남'] > 0].iloc[1:3])  # 1번째부터 2번째까지 행
print(df[df['강남'] > 0].iloc[1:3, 1:3])  # 행과 열 모두 슬라이싱
print(df[df['강남'] > 0].iloc[1:3, [1, 2]])  # 특정 열 인덱스로 선택
print(df[df['강남'] > 0].iloc[1:3, [0, 2]])  # 0번째, 2번째 열 선택
print(df[df['강남'] > 0].iloc[1:3, [0, 1]])  # 0번째, 1번째 열 선택
```

## 10.3 주요 메서드 정리

### 재색인 메서드 옵션
- `fill_value`: 결측값을 특정 값으로 채우기
- `method='ffill'`: 이전 값으로 채우기 (forward fill)
- `method='backfill'`: 다음 값으로 채우기 (backward fill)

### 인덱싱 방법
- **열 선택**: `df['열이름']`
- **불리언 인덱싱**: `df[조건문]`
- **loc**: 라벨 기반 인덱싱 `df.loc[행, 열]`
- **iloc**: 정수 위치 기반 인덱싱 `df.iloc[행위치, 열위치]`

## 10.4 실무 활용 팁

### 조건부 데이터 선택 패턴
```python
# 여러 조건 조합
condition = (df['강남'] > 0) & (df['강북'] < 10)
selected_data = df[condition][['강남', '서초']]

# 조건에 따른 데이터 수정
df.loc[df['강남'] > 5, '강남'] = 999
```

### 결측값 처리 전략
- 작은 데이터셋: `fill_value`로 특정값 채우기
- 시계열 데이터: `ffill` 또는 `backfill` 사용
- 분석 목적에 따라 적절한 방법 선택

