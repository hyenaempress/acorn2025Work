# NumPy 실습 문제 해답 비교 및 최적화 가이드

## 개요

NumPy 실습 문제에 대한 두 가지 접근 방식을 비교하고, NumPy다운 코드 작성법을 학습하는 가이드입니다.

## Step 1: 정규분포 난수 배열 생성 및 통계

### 문제
정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.

### 💡 최적화된 답안
```python
import numpy as np

print("--- Step1: 정규분포 난수 5x4 배열, 행별 합계/최댓값 ---")
arr = np.random.randn(5, 4)

# enumerate를 활용한 깔끔한 반복
for i, row in enumerate(arr):
    print(f"{i+1}행 합계   : {row.sum()}")
    print(f"{i+1}행 최댓값 : {row.max()}")
```

### 📊 일반적인 답안
```python
data = np.random.randn(5, 4)
i = 1
for row in data:
    print('행합계:', np.sum(row))
    print('행최대값:', np.max(row))
    print('행최소값:', np.min(row))
```

### 🔍 차이점 분석
| 항목 | 최적화된 답안 | 일반적인 답안 |
|------|---------------|---------------|
| 인덱싱 | `enumerate()` 사용 | 수동 카운터 |
| 출력 형식 | f-string 사용 | 기본 문자열 |
| 메서드 호출 | `row.sum()` | `np.sum(row)` |

## Step 2-1: 6×6 배열 인덱싱

### 문제
6행 6열의 다차원 zero 행렬 객체를 생성한 후 1~36까지 정수로 채우고 인덱싱하시오.

### 💡 최적화된 답안 (핵심 차이점!)
```python
print("--- Step2-1: 6x6 zero 행렬, indexing ---")
arr = np.zeros((6, 6))

# ✅ 벡터화 연산 - 한 줄로 해결!
arr.flat[:] = np.arange(1, 37)

print("2번째 행 전체:", arr[1])
print("5번째 열 전체:", arr[:, 4])
print("15~29 추출:")
print(arr[2:5, 2:5])
```

### 📊 일반적인 답안
```python
zarr = np.zeros((6,6))
cnt = 0
# ❌ 비효율적인 이중 반복문
for i in range(6):
    for j in range(6):
        cnt += 1
        zarr[i, j] = cnt
```

### ⚡ 성능 비교
```python
import time

# 벡터화 방법 (최적화됨)
start = time.time()
arr = np.zeros((6, 6))
arr.flat[:] = np.arange(1, 37)
time1 = time.time() - start

# 반복문 방법 (일반적)
start = time.time()
zarr = np.zeros((6, 6))
cnt = 0
for i in range(6):
    for j in range(6):
        cnt += 1
        zarr[i, j] = cnt
time2 = time.time() - start

print(f"벡터화: {time1:.6f}초")
print(f"반복문: {time2:.6f}초")
print(f"성능 향상: {time2/time1:.1f}배")
```

## Step 2-2: 6×4 배열 초기화 및 수정

### 문제
6행 4열 배열에 난수로 시작하여 1씩 증가하는 값으로 채우고, 첫 행과 마지막 행을 수정하시오.

### 💡 최적화된 답안
```python
print("--- Step2-2: 6x4 zero 행렬, 난수로 초기화 및 수정 ---")
arr = np.zeros((6, 4))
rand_start = np.random.randint(20, 101, 6)

# ✅ 벡터화된 행 단위 초기화
for i in range(6):
    arr[i] = np.arange(rand_start[i], rand_start[i]+4)

print("초기화 완료:")
print(arr)

# 첫 행과 마지막 행 수정
arr[0] = 1000
arr[-1] = 6000
print("수정 완료:")
print(arr)
```

### 📊 일반적인 답안
```python
zarr = np.zeros((6,4))
ran = np.random.randint(20, 100, 6)
ran = list(ran)  # ❌ 불필요한 리스트 변환

# ❌ 비효율적인 이중 반복문 + pop()
for row in range(len(zarr)):
    num = ran.pop(0)
    for col in range(len(zarr[0])):
        zarr[row][col] = num
        num += 1
```

### 🔍 주요 차이점
1. **데이터 처리**: `np.arange()` vs 이중 반복문
2. **메모리 효율성**: 벡터 연산 vs 개별 할당
3. **코드 가독성**: 간결함 vs 복잡함

## Step 3: 기술통계량 계산

### 문제
4행 5열 정규분포 배열의 기술통계량을 구하시오.

### 💡 공통 답안 (둘 다 적절함)
```python
print("--- Step3: 4x5 정규분포 배열, 기술통계량 ---")
arr = np.random.randn(4, 5)

print("평균:", np.mean(arr))
print("합계:", np.sum(arr))
print("표준편차:", np.std(arr))
print("분산:", np.var(arr))
print("최댓값:", np.max(arr))
print("최솟값:", np.min(arr))
print("1사분위수:", np.percentile(arr, 25))
print("2사분위수:", np.percentile(arr, 50))  # 중앙값
print("3사분위수:", np.percentile(arr, 75))
print("요소값 누적합:", np.cumsum(arr))
```

## 추가 문제들 해답 비교

### Q1: 브로드캐스팅과 조건 연산

#### 📋 문제
다음 두 배열이 있을 때, 두 배열을 브로드캐스팅하여 곱한 결과를 출력하고, 그 결과에서 값이 30 이상인 요소만 골라 출력하시오.

#### 💡 최적화된 답안
```python
print("--- Q1: 브로드캐스팅과 조건 연산 ---")
a = np.array([[1], [2], [3]])  # 3x1 배열
b = np.array([10, 20, 30])     # 1x3 배열

# 브로드캐스팅으로 3x3 결과 생성
result = a * b
print("곱한 결과:")
print(result)
# [[10 20 30]
#  [20 40 60]  
#  [30 60 90]]

# 불린 인덱싱으로 조건부 추출
filtered = result[result >= 30]
print("30 이상인 요소:", filtered)
```

#### 📊 일반적인 답안
```python
result = a * b
print('곱한 결과 : ', result)
print('30 이상인 요소만 골라 출력 : ', result[result >= 30])
```

#### 🔍 차이점 분석
- **구조**: 최적화된 답안이 더 체계적으로 단계별 설명
- **출력**: 결과 해석을 위한 주석 추가
- **기능**: 동일하게 올바른 브로드캐스팅과 불린 인덱싱 사용

---

### Q2: 다차원 배열 슬라이싱 및 재배열

#### 📋 문제
3×4 크기의 배열을 만들고, 2번째 행과 1번째 열을 출력한 후, (4, 3) 형태로 reshape하고 flatten으로 1차원 배열 만들기

#### 💡 최적화된 답안
```python
print("--- Q2: 슬라이싱 및 재배열 ---")
arr = np.arange(1, 13).reshape(3, 4)
print("원본 배열:")
print(arr)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

print("2번째 행:", arr[1])      # [5 6 7 8]
print("1번째 열:", arr[:, 0])   # [1 5 9]

# 재배열
reshaped = arr.reshape(4, 3)
print("(4,3)로 reshape:")
print(reshaped)

print("flatten 결과:", reshaped.flatten())
```

#### 📊 일반적인 답안
```python
arr = np.arange(1, 13).reshape(3, 4)
print('arr : ', arr)
print('2번째 행 전체 출력 : ', arr[1])
print('1번째 열 전체 출력 : ', arr[:, 0])
reshaped = arr.reshape(4, 3)
print(reshaped.flatten())
```

#### 🔍 차이점 분석
- **가독성**: 최적화된 답안이 각 단계별 결과를 명확히 표시
- **설명**: 주석으로 예상 결과값 제공
- **구조**: 논리적 순서로 단계별 진행

---

### Q3: 복합 조건과 배열 연산

#### 📋 문제
1부터 100까지의 수에서 3의 배수이면서 5의 배수가 아닌 값을 추출하고, 그 값들을 제곱한 배열을 만들어 출력하시오.

#### 💡 최적화된 답안
```python
print("--- Q3: 조건 추출 및 제곱 ---")
arr = np.arange(1, 101)

# 복합 조건: 3의 배수 AND 5의 배수가 아님
condition = (arr % 3 == 0) & (arr % 5 != 0)
filtered = arr[condition]

print("3의 배수이면서 5의 배수가 아닌 값:")
print(filtered)
# [ 3  6  9 12 18 21 24 27 33 36 39 42 48 51 54 57 63 66 69 72 78 81 84 87 93 96 99]

# 벡터화된 제곱 연산
squared = filtered ** 2
print("제곱한 배열:")
print(squared)
```

#### 📊 일반적인 답안
```python
arr = np.arange(1, 101)
imsi = (arr % 3 == 0) & (arr % 5 != 0)
filtered = arr[imsi]
print('3의 배수이면서 5의 배수가 아닌 값 : ', filtered)
squared = filtered ** 2
print('제곱한 배열 : ', squared)
```

#### 🔍 차이점 분석
- **변수명**: `condition` vs `imsi` (의미있는 변수명 사용)
- **설명**: 조건의 의미를 주석으로 명확히 설명
- **출력**: 중간 결과를 보여주어 이해도 향상

---

### Q4: 조건부 변환과 배열 복사

#### 📋 문제
주어진 배열에서 10 이상이면 'High', 그렇지 않으면 'Low'로 변환하고, 20 이상인 요소를 -1로 바꾼 새로운 배열을 만들어 출력하시오 (원본 유지).

#### 💡 최적화된 답안
```python
print("--- Q4: 조건에 따른 변환 ---")
arr = np.array([15, 22, 8, 19, 31, 4])
print("원본 배열:", arr)

# 1. 문자열 배열로 변환
labels = np.where(arr >= 10, 'High', 'Low')
print("문자열 변환 결과:", labels)
# ['High' 'High' 'Low' 'High' 'High' 'Low']

# 2. 원본 보존하며 값 수정
new_arr = arr.copy()  # 🔑 중요: 원본 보존!
new_arr[new_arr >= 20] = -1
print("20 이상 → -1 변환:", new_arr)
print("원본 배열 (보존됨):", arr)
```

#### 📊 일반적인 답안
```python
arr = np.array([15, 22, 8, 19, 31, 4])
labels = np.where(arr >= 10, 'High', 'Low')
print('문자열 배열 : ', labels)

new_arr = np.copy(arr)
print('new_arr : ', new_arr)
new_arr[new_arr >= 20] = -1
print('new_arr : ', new_arr)
```

#### 🔍 차이점 분석
- **copy 방법**: `arr.copy()` vs `np.copy(arr)` (둘 다 정확)
- **검증**: 원본 보존 확인을 위한 출력 추가
- **설명**: 각 단계의 목적을 명확히 설명

---

### Q5: 정규분포와 백분위수

#### 📋 문제
정규분포(평균 50, 표준편차 10)를 따르는 난수 1000개를 만들고, 상위 5% 값만 출력하세요.

#### 💡 최적화된 답안
```python
print("--- Q5: 정규분포 난수, 상위 5% ---")

# 정규분포 난수 생성 (평균=50, 표준편차=10)
data = np.random.normal(loc=50, scale=10, size=1000)

# 상위 5% 경계값 = 95백분위수
threshold = np.percentile(data, 95)
top5_percent = data[data >= threshold]

print(f"생성된 데이터 통계:")
print(f"  평균: {np.mean(data):.2f}")
print(f"  표준편차: {np.std(data):.2f}")
print(f"95백분위수 (상위 5% 경계): {threshold:.2f}")
print(f"상위 5% 데이터 개수: {len(top5_percent)}")
print(f"상위 5% 평균값: {np.mean(top5_percent):.2f}")
print(f"상위 5% 값들: {top5_percent}")
```

#### 📊 일반적인 답안
```python
data = np.random.normal(loc=50, scale=10, size=1000)
threshold = np.percentile(data, 95)  # 상위 5% 경계값
top5 = data[data > threshold]
print('상위 5% 값 : ', threshold)
print('상위 5% 데이터 수 : ', len(top5))
```

#### 🔍 차이점 분석
- **조건**: `>=` vs `>` (경계값 포함 여부)
- **분석 깊이**: 추가 통계 정보 제공
- **출력 형식**: f-string 사용으로 가독성 향상
- **해석**: 결과에 대한 통계적 해석 추가

## 📊 문제별 난이도 및 핵심 개념

| 문제 | 난이도 | 핵심 개념 | 주요 함수/기법 |
|------|--------|-----------|----------------|
| Q1 | ⭐⭐ | 브로드캐스팅, 불린 인덱싱 | `*`, `[]` |
| Q2 | ⭐ | 슬라이싱, reshape | `[:,:]`, `.reshape()`, `.flatten()` |  
| Q3 | ⭐⭐⭐ | 복합 조건, 불린 인덱싱 | `&`, `%`, `**` |
| Q4 | ⭐⭐ | 조건부 변환, 배열 복사 | `np.where()`, `.copy()` |
| Q5 | ⭐⭐⭐ | 정규분포, 백분위수 | `np.random.normal()`, `np.percentile()` |

## 🎯 문제별 학습 포인트

### Q1에서 배울 것
- **브로드캐스팅**: (3,1) × (3,) → (3,3) 자동 확장
- **불린 인덱싱**: 조건을 만족하는 요소만 추출

### Q2에서 배울 것  
- **슬라이싱**: `[행, 열]` 형태의 인덱싱
- **배열 변형**: `reshape()`과 `flatten()`의 차이

### Q3에서 배울 것
- **복합 조건**: `&` (AND), `|` (OR) 연산자 사용
- **조건 우선순위**: 괄호로 조건 묶기의 중요성

### Q4에서 배울 것
- **원본 보존**: `.copy()`의 중요성
- **조건부 변환**: `np.where()`의 다양한 활용

### Q5에서 배울 것
- **정규분포 생성**: `loc`(평균), `scale`(표준편차) 매개변수
- **백분위수**: 상위 5% = 95백분위수의 개념

## 🚀 NumPy 최적화 핵심 원칙

### 1. 벡터화 우선 (Vectorization First!)
```python
# ❌ Python 스타일 (느림)
result = []
for i in range(len(arr)):
    result.append(arr[i] * 2)

# ✅ NumPy 스타일 (빠름)
result = arr * 2
```

### 2. 브로드캐스팅 활용
```python
# ❌ 반복문으로 각 행에 더하기
for i in range(len(matrix)):
    matrix[i] = matrix[i] + vector

# ✅ 브로드캐스팅으로 한 번에
matrix = matrix + vector
```

### 3. 불린 인덱싱 활용
```python
# ❌ 반복문으로 조건 확인
result = []
for x in arr:
    if x > threshold:
        result.append(x)

# ✅ 불린 인덱싱으로 한 번에
result = arr[arr > threshold]
```

### 4. 유용한 NumPy 메서드들
```python
# 배열 초기화
arr.flat[:] = values           # 1차원으로 접근하여 초기화
arr.fill(value)               # 모든 요소를 같은 값으로

# 배열 조작
np.where(condition, x, y)     # 조건부 선택
np.clip(arr, min, max)        # 값 범위 제한
arr.copy()                    # 원본 보존 복사

# 통계 함수
np.percentile(arr, q)         # 백분위수
np.cumsum(arr)               # 누적합
np.cumprod(arr)              # 누적곱
```

## 📊 성능 비교 요약

### 시간 복잡도
| 방법 | 시간 복잡도 | 메모리 사용 | 가독성 |
|------|-------------|-------------|--------|
| 벡터화 연산 | O(1) | 효율적 | 높음 |
| 단일 반복문 | O(n) | 보통 | 보통 |
| 이중 반복문 | O(n²) | 비효율적 | 낮음 |

### 실제 성능 테스트
```python
import numpy as np
import time

# 큰 배열로 성능 테스트
size = 1000

# 벡터화 방법
start = time.time()
arr1 = np.zeros((size, size))
arr1.flat[:] = np.arange(size * size)
time1 = time.time() - start

# 반복문 방법
start = time.time()
arr2 = np.zeros((size, size))
cnt = 0
for i in range(size):
    for j in range(size):
        arr2[i, j] = cnt
        cnt += 1
time2 = time.time() - start

print(f"벡터화: {time1:.4f}초")
print(f"반복문: {time2:.4f}초")
print(f"성능 향상: {time2/time1:.0f}배")
```

## 🎯 학습 로드맵

### 초급 → 중급
1. **기본 반복문** → **벡터화 연산**
2. **개별 할당** → **배열 연산**
3. **Python 스타일** → **NumPy 스타일**

### 중급 → 고급
1. **단순 조건** → **복합 조건과 불린 인덱싱**
2. **기본 통계** → **고급 통계 함수**
3. **메모리 무시** → **메모리 효율성 고려**

## 📚 추천 학습 순서

1. **NumPy 기본 개념** 숙지
2. **벡터화 연산** 이해 및 적용
3. **브로드캐스팅** 활용법 학습
4. **불린 인덱싱** 마스터
5. **성능 최적화** 기법 적용

## 🔧 실무 팁

### 디버깅 팁
```python
# 배열 형태 확인
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Size: {arr.size}")

# 조건 확인
condition = arr > 0
print(f"True 개수: {np.sum(condition)}")
print(f"False 개수: {len(arr) - np.sum(condition)}")
```

### 메모리 사용량 확인
```python
# 메모리 사용량 (바이트)
print(f"메모리 사용량: {arr.nbytes} bytes")
print(f"요소당 크기: {arr.itemsize} bytes")
```

## 결론

**핵심 메시지**: "NumPy의 진정한 힘은 벡터화에 있습니다!"

- **반복문을 피하고** 배열 연산을 활용하세요
- **브로드캐스팅을** 이해하고 적극 활용하세요  
- **불린 인덱싱으로** 조건부 처리를 간단하게 하세요
- **원본 데이터 보존**을 위해 `.copy()`를 사용하세요

NumPy다운 코드를 작성하면 **성능은 10-100배 향상**되고, **코드는 더 간결**해집니다!