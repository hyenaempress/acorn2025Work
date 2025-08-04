# Pandas

**Pandas**는 행과 열이 단순 정수형 인덱스가 아닌 레이블로 식별되는 데이터 구조를 제공하는 라이브러리로, NumPy의 구조화된 배열들을 보완(강화)한 모듈이다.

## 주요 특징
- **고수준의 자료구조** 제공: 시계열 축약연산, 누락데이터 처리, SQL, 시각화 기능
- **핵심 데이터 구조**: Series, DataFrame
- **데이터 분석 도구** 제공

## 기본 Import
```python
import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np
```

> **참고**: Pandas는 NumPy를 기반으로 하며, NumPy의 배열을 Pandas의 데이터 구조로 변환할 수 있다.

## Series

**Series**는 일련의 객체를 담을 수 있는 1차원 배열과 유사한 구조로, 인덱스와 값으로 구성된 데이터 구조이다. 인덱스 색인을 가지며, list, array, dict 등 다양한 형태의 데이터를 담을 수 있다.

### 기본 Series 생성

```python
# 리스트로 Series 생성
obj = Series([3, 7, -5, 4])
print(obj)
# 출력:
# 0    3
# 1    7
# 2   -5
# 3    4
# dtype: int64

# 튜플로도 생성 가능
obj = Series((3, 7, -5, 4))
print(obj, type(obj))
```

**특징**: 
- 인덱싱이 자동으로 따라 붙는다
- 자동 인덱싱 라벨이 명시적이다 (NumPy는 묵시적)
- 리스트와 튜플은 순서가 있어 가능하지만 Set은 순서가 없기 때문에 불가능하다

### 사용자 정의 인덱스

```python
# 인덱스를 문자열로 지정
obj2 = Series([3, 7, -5, 4], index=['a', 'b', 'c', 'd'])
print(obj2)
# 출력:
# a    3
# b    7
# c   -5
# d    4
# dtype: int64
```

### Series 연산 및 속성

```python
# 합계 계산
print(obj2.sum(), np.sum(obj2))  # 급할 때는 파이썬의 sum() 함수 사용 가능

# 값들만 출력
print(obj2.values)  # [ 3  7 -5  4]

# 인덱스들만 출력
print(obj2.index)   # Index(['a', 'b', 'c', 'd'], dtype='object')
```

### Series 인덱싱 및 슬라이싱

```python
# 단일 값 접근
print(obj2[1])      # 정수 인덱스로 접근: 7
print(obj2['a'])    # 라벨 인덱스로 접근 (값만 반환): 3

# Series 형태로 반환
print(obj2[['a']])  # Series 형태로 반환 (차원 형태 유지)

# 슬라이싱
print(obj2[1:3])    # 정수 인덱스 슬라이싱
print(obj2['a':'d']) # 라벨 인덱스 슬라이싱

# 여러 인덱스 접근
print(obj2[['a', 'd']])  # 여러 인덱스로 값 접근 (리스트 형태)
```

### 인덱싱 방법 정리

| 방법 | 설명 | 반환 형태 |
|------|------|-----------|
| `obj2['a']` | 단일 라벨로 접근 | 값만 반환 |
| `obj2[['a']]` | 리스트 형태로 접근 | Series 형태 반환 |
| `obj2[['a', 'd']]` | 여러 라벨 접근 | Series 형태 반환 |
| `obj2['a':'d']` | 라벨 슬라이싱 | Series 형태 반환 |

### 위치 기반 인덱싱 (iloc)

```python
# iloc를 사용한 위치 기반 접근
print(obj2[3])              # 일반 위치 접근
print(obj2.iloc[3])         # iloc를 사용한 위치 접근
print(obj2.iloc[[2,3]])     # 여러 위치 접근
```

### 불리언 인덱싱

```python
# 조건에 따른 데이터 필터링
print(obj2 > 0)             # 불리언 마스크 생성
print(obj2[obj2 > 0])       # 0보다 큰 값들만 선택

# 인덱스 존재 여부 확인
print('a' in obj2)          # True
print('aa' in obj2)         # False
```

### 딕셔너리로 Series 만들기

딕셔너리로 Series를 만들면 키가 인덱스가 되고, 값이 Series의 값이 된다.

```python
# 딕셔너리로 Series 생성
data = {'a': 3, 'b': 7, 'c': -5, 'd': 4}
obj3 = Series(data)
print(obj3)

# 실제 예제
names = {'mouse': 5000, 'keyboard': 10000, 'monitor': 20000}
obj3 = Series(names)
print(obj3)

# 인덱스 변경
obj3.index = ['마우스', '키보드', '모니터']
print(obj3)
print(obj3['마우스'])       # 5000

# Series에 이름 부여
obj3.name = '전자제품 가격'
print(obj3)
```

> **참고**: 파이썬 초기 버전에서는 딕셔너리도 순서가 없었지만, 현재는 순서가 보장된다.

## DataFrame

**DataFrame**은 2차원 테이블 형태의 데이터 구조로, 행과 열로 구성되며 각 열은 Series 형태로 저장된다. DataFrame은 Series 객체들이 모여 표를 구성하는 것이다.

### Series로 DataFrame 생성

```python
# Series를 사용해 DataFrame 생성
df = DataFrame(obj3)
print(df)
```

### 딕셔너리로 DataFrame 생성

```python
# 딕셔너리로 DataFrame 생성
data = {
    'irum': ['홍길동', '한국인', '신기해', '공기밥', '한가해'],
    'juso': ('역삼동', '신당동', '역삼동', '역삼동', '신사동'),
    'nai': [23, 25, 33, 30, 35],
}

frame = DataFrame(data)
print(frame)
```

### DataFrame 컬럼 접근

```python
# 컬럼 선택 방법
print(frame['irum'])        # 'irum' 열 선택 (대괄호 방식)
print(frame.irum)          # 'irum' 열 선택 (속성 방식)
print(type(frame.irum))    # <class 'pandas.core.series.Series'>
```

### 열 순서 변경

```python
# 열 순서를 지정하여 DataFrame 생성
print(DataFrame(data, columns=['juso', 'irum', 'nai']))
```

### NaN 값이 포함된 DataFrame

```python
# 존재하지 않는 컬럼 추가 시 NaN 값 생성
frame2 = DataFrame(data, 
                   columns=['irum', 'nai', 'juso', 'tel'],
                   index=['a', 'b', 'c', 'd', 'e'])
print(frame2)
```

**결과**:
```
    irum  nai  juso   tel
a   홍길동   23  역삼동   NaN
b   한국인   25  신당동   NaN
c   신기해   33  역삼동   NaN
d   공기밥   30  역삼동   NaN
e   한가해   35  신사동   NaN
```

### DataFrame 컬럼에 값 부여

DataFrame의 컬럼에 값을 부여하는 다양한 방법들이 있다:

#### 1. 개별 값들로 리스트 할당
```python
# 'tel' 열에 개별 값들을 리스트로 할당
frame2['tel'] = ['010-1234-5678', '02-9876-5432', np.nan, '031-1111-2222', '02-3333-4444']
print(frame2)
```

**결과**:
```
    irum  nai  juso          tel
a   홍길동   23  역삼동  010-1234-5678
b   한국인   25  신당동   02-9876-5432
c   신기해   33  역삼동           NaN
d   공기밥   30  역삼동  031-1111-2222
e   한가해   35  신사동   02-3333-4444
```

#### 2. 모든 행에 동일한 값 할당 (브로드캐스팅)
```python
# 모든 행에 동일한 값 추가 (하나로 밀어넣기)
frame2['tel'] = '111-1111-1111'
print(frame2)
```

**결과**:
```
    irum  nai  juso         tel
a   홍길동   23  역삼동  111-1111-1111
b   한국인   25  신당동  111-1111-1111
c   신기해   33  역삼동  111-1111-1111
d   공기밥   30  역삼동  111-1111-1111
e   한가해   35  신사동  111-1111-1111
```

#### 3. Series 객체로 할당
```python
# Series 객체를 사용하여 특정 인덱스에만 값 할당
val = Series(['010-1234-5678', '02-9876-5432', np.nan], index=['a', 'b', 'c'])
frame2['tel'] = val
print(frame2)
```

**결과**:
```
    irum  nai  juso          tel
a   홍길동   23  역삼동  010-1234-5678
b   한국인   25  신당동   02-9876-5432
c   신기해   33  역삼동           NaN
d   공기밥   30  역삼동           NaN  # 할당되지 않은 행은 NaN
e   한가해   35  신사동           NaN  # 할당되지 않은 행은 NaN
```

> **참고**: Series로 할당할 때는 인덱스가 일치하는 행에만 값이 할당되고, 나머지는 NaN이 된다.

### DataFrame 전치 (Transpose)

```python
# 데이터 프레임 전치 (행과 열을 바꿈)
print(frame2.T)
```

**결과**:
```
        a      b      c      d      e
irum   홍길동    한국인    신기해    공기밥    한가해
nai     23     25     33     30     35
juso   역삼동    신당동    역삼동    역삼동    신사동
tel    010-1234-5678  02-9876-5432  NaN  031-1111-2222  02-3333-4444
```

> **참고**: `.T` 속성을 사용하면 행과 열이 바뀌어 출력된다. 원본 데이터는 변경되지 않는다.

### DataFrame 값 추출

```python
# 데이터 프레임의 값들만 출력 (2차원 배열로 변환)
print(frame2.values, type(frame2.values))
# 결과: <class 'numpy.ndarray'>
```

> **참고**: `.values`를 사용하면 DataFrame이 NumPy 2차원 배열로 변환된다. 나중에 사용할 일이 많다.

### 행/열 삭제

```python
# 행 삭제 (인덱스가 'd'인 행 삭제)
frame3 = frame2.drop('d')
print(frame3)

# 열 삭제 (열이름이 'tel'인 열 삭제)
frame4 = frame2.drop('tel', axis=1)
print(frame4)
```

### 데이터 프레임 정렬 (Sort)

```python
# 행 인덱스 기준으로 내림차순 정렬
print(frame2.sort_index(axis=0, ascending=False))

# 열 인덱스 기준으로 내림차순 정렬
print(frame2.sort_index(axis=1, ascending=False))

# 열 인덱스 기준으로 오름차순 정렬
print(frame2.sort_index(axis=1, ascending=True))

# 값의 개수 세기
print(frame['juso'].value_counts())  # 'juso' 열의 값 개수 세기

# 특정 열 값을 기준으로 정렬
print(frame2.sort_values(by='nai', ascending=True))   # 'nai' 열 기준 오름차순
print(frame2.sort_values(by='nai', ascending=False))  # 'nai' 열 기준 내림차순
```

> **참고**: Sort 알고리즘을 몰라도 데이터 분석이 가능하지만, 알고리즘을 이해하는 것이 중요하다. `value_counts()`를 모르면 for문을 돌려서 누적해야 한다.

### 문자열 처리

```python
# 문자열 자르기 예제
data = {
    'juso': ['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon': [23, 25, 15],
}
fr = pd.DataFrame(data)
print(fr)

# 문자열을 공백으로 분리하여 첫 번째와 두 번째 부분 추출
result1 = Series([x.split()[0] for x in fr.juso])  # 첫 번째 부분 (구)
result2 = Series([x.split()[1] for x in fr.juso])  # 두 번째 부분 (동)

print(result1)                    # 첫 번째 부분 출력
print(result2)                    # 두 번째 부분 출력
print(result1.value_counts())     # 첫 번째 부분의 값 개수 세기
print(result2.value_counts())     # 두 번째 부분의 값 개수 세기
```

**결과 예시**:
```
# result1 출력
0    강남구
1     중구
2    강남구
dtype: object

# result1.value_counts() 출력
강남구    2
중구     1
dtype: int64
```

> **참고**: 
> - Series는 리스트 컴프리헨션으로 생성할 수 있고, 자동으로 0, 1, 2... 숫자 인덱스를 가진다
> - 이러한 인덱스를 이용해서 데이터를 효율적으로 다룰 수 있다
> - 빅데이터 분석에서 이런 기능들이 매우 중요하게 된다

## 예제

```python
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# 전체 예제
data = Series([10, 20, 30, 40], index=['서울', '부산', '대구', '인천'])
print("Series 데이터:")
print(data)
print(f"합계: {data.sum()}")
print(f"평균: {data.mean()}")

# DataFrame 예제
df_data = {
    '도시': ['서울', '부산', '대구', '인천'],
    '인구': [970, 340, 250, 290],
    '면적': [605, 770, 884, 1063]
}
df = DataFrame(df_data)
print("\nDataFrame 데이터:")
print(df)
```

위의 방법들로 Series와 DataFrame을 효율적으로 운영할 수 있다.