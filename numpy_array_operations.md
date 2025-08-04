# NumPy 배열 조작과 연산 완전 가이드

## 개요

NumPy 배열에서 행과 열을 추가하거나 삭제하는 다양한 방법과 조건 연산에 대한 완전 가이드입니다.

## 1. 배열 구조 변경 기본 함수

### 1.1 `np.c_`와 `np.r_` - 배열 연결

```python
import numpy as np

# 3x3 단위 행렬 생성
aa = np.eye(3)
print("원본 배열 aa:")
print(aa)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# np.c_: 열 방향으로 연결 (column-wise)
bb = np.c_[aa, aa[2]]  # aa의 2번째 행을 열로 추가
print("열 추가 (np.c_):")
print(bb)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 1.]]

# np.r_: 행 방향으로 연결 (row-wise)
cc = np.r_[aa, [aa[2]]]  # aa의 2번째 행을 행으로 추가
print("행 추가 (np.r_):")
print(cc)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
```

### 1.2 차원 변경 효과

```python
# 1차원 배열을 2차원으로 변환
a = np.array([1, 2, 3])
print("원본 1차원:", a)              # [1 2 3]

# np.c_를 사용한 차원 변환
print("np.c_로 변환:")
print(np.c_[a])                     # [[1] [2] [3]] - 3x1 행렬
print("np.c_[a].shape:", np.c_[a].shape)  # (3, 1)

# reshape을 사용한 차원 변환 (동일한 결과)
print("reshape(3, 1):")
print(a.reshape(3, 1))              # [[1] [2] [3]]
```

## 2. 배열 요소 추가/삽입/삭제

### 2.1 append - 배열 끝에 추가

```python
print("=== append 연산 ===")
a = np.array([1, 2, 3])

# 1차원 배열에 요소 추가
b = np.append(a, [4, 5])
print("append 결과:", b)            # [1 2 3 4 5]
```

### 2.2 insert - 특정 위치에 삽입

```python
print("=== insert 연산 ===")
# 인덱스 2에 [6, 7] 삽입
c = np.insert(b, 2, [6, 7])
print("insert 결과:", c)            # [1 2 6 7 3 4 5]

# ⚠️ 주의: insert는 차원을 축소시킴
aa = np.arange(1, 10).reshape(3, 3)
print("원본 2차원 배열:")
print(aa)

# 차원 축소됨 (2차원 → 1차원)
result = np.insert(aa, 1, 99)
print("insert (차원 축소):", result)
print("결과 shape:", result.shape)   # (10,) - 1차원으로 축소!
```

### 2.3 axis 매개변수로 차원 유지

```python
print("=== axis를 사용한 차원 유지 ===")
aa = np.arange(1, 10).reshape(3, 3)

# axis=0: 행 방향으로 삽입 (차원 유지)
result_row = np.insert(aa, 1, 99, axis=0)
print("행 삽입 (axis=0):")
print(result_row)
print("shape:", result_row.shape)    # (4, 3)

# axis=1: 열 방향으로 삽입 (차원 유지)
result_col = np.insert(aa, 1, 99, axis=1)
print("열 삽입 (axis=1):")
print(result_col)
print("shape:", result_col.shape)    # (3, 4)
```

## 3. 2차원 배열에서의 추가 연산

### 3.1 행 추가

```python
print("=== 2차원 배열 행 추가 ===")
aa = np.arange(1, 10).reshape(3, 3)

# 방법 1: append with axis=0
new_row = np.append(aa, [[99, 99, 99]], axis=0)
print("행 추가 결과:")
print(new_row)
# [[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]
#  [99 99 99]]

# 방법 2: 여러 행 추가
bb = np.arange(10, 19).reshape(3, 3)
combined = np.append(aa, bb, axis=0)
print("여러 행 추가:")
print(combined)
```

### 3.2 열 추가

```python
print("=== 2차원 배열 열 추가 ===")
# 열 추가 시 올바른 형태로 제공
new_col = np.append(aa, [[88], [88], [88]], axis=1)
print("열 추가 결과:")
print(new_col)
# [[ 1  2  3 88]
#  [ 4  5  6 88]
#  [ 7  8  9 88]]
```

## 4. 배열 요소 삭제

### 4.1 delete 함수

```python
print("=== delete 연산 ===")
aa = np.arange(1, 10).reshape(3, 3)

# 차원 축소됨 (기본 동작)
deleted = np.delete(aa, 1)
print("delete (차원 축소):", deleted)
print("shape:", deleted.shape)       # (8,) - 1차원으로 축소

# axis=0: 행 삭제 (차원 유지)
delete_row = np.delete(aa, 1, axis=0)
print("행 삭제 (axis=0):")
print(delete_row)
print("shape:", delete_row.shape)    # (2, 3)

# axis=1: 열 삭제 (차원 유지)
delete_col = np.delete(aa, 1, axis=1)
print("열 삭제 (axis=1):")
print(delete_col)
print("shape:", delete_col.shape)    # (3, 2)
```

### 4.2 여러 요소 삭제

```python
# 여러 인덱스 삭제
multiple_delete = np.delete(aa, [1, 2], axis=0)
print("여러 행 삭제:", multiple_delete)
```

## 5. 조건 연산 (where)

### 5.1 기본 사용법

```python
print("=== 조건 연산 (where) ===")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
condition = np.array([True, False, True])

# where(조건, 참일_때_값, 거짓일_때_값)
result = np.where(condition, x, y)
print("조건 배열:", condition)
print("x 배열:", x)
print("y 배열:", y)
print("where 결과:", result)         # [1 5 3]
```

### 5.2 수치 조건 활용

```python
print("=== 수치 조건 활용 ===")
arr = np.array([1, 2, 3, 4, 5, 6])

# 3보다 큰 값은 그대로, 작거나 같은 값은 0으로
result = np.where(arr > 3, arr, 0)
print("원본:", arr)
print("arr > 3인 경우:", result)     # [0 0 0 4 5 6]

# 짝수는 그대로, 홀수는 -1로
result2 = np.where(arr % 2 == 0, arr, -1)
print("짝수 조건:", result2)         # [-1  2 -1  4 -1  6]
```

### 5.3 2차원 배열에서의 조건 연산

```python
print("=== 2차원 배열 조건 연산 ===")
matrix = np.arange(1, 13).reshape(3, 4)
print("원본 행렬:")
print(matrix)

# 6보다 큰 값은 그대로, 작거나 같은 값은 0으로
result = np.where(matrix > 6, matrix, 0)
print("조건 적용 결과:")
print(result)
```

## 6. 차원 축소/확장 주의사항

### 6.1 함수별 차원 변화 정리

| 함수 | 기본 동작 | axis 미지정 | axis 지정 |
|------|----------|-------------|-----------|
| `np.insert()` | 차원 축소 | 1차원 배열 반환 | 차원 유지 |
| `np.append()` | 차원 축소 | 1차원 배열 반환 | 차원 유지 |
| `np.delete()` | 차원 축소 | 1차원 배열 반환 | 차원 유지 |

### 6.2 차원 유지를 위한 권장 사항

```python
print("=== 차원 유지 권장 방법 ===")
aa = np.arange(1, 10).reshape(3, 3)

# ❌ 차원이 축소됨
bad_result = np.append(aa, [99, 99, 99])
print("나쁜 예 (차원 축소):", bad_result.shape)  # (12,)

# ✅ 차원이 유지됨
good_result = np.append(aa, [[99, 99, 99]], axis=0)
print("좋은 예 (차원 유지):", good_result.shape)  # (4, 3)
```

## 7. 실무 활용 예제

### 7.1 데이터 전처리에서의 활용

```python
print("=== 실무 활용 예제 ===")
# 가상의 학생 성적 데이터
scores = np.array([[85, 90, 78],
                   [92, 88, 95],
                   [76, 82, 89]])

print("원본 성적 데이터:")
print("수학 영어 과학")
print(scores)

# 평균 점수 열 추가
averages = np.mean(scores, axis=1).reshape(-1, 1)
scores_with_avg = np.append(scores, averages, axis=1)
print("\n평균 점수 추가:")
print("수학 영어 과학 평균")
print(scores_with_avg)

# 조건부 등급 부여 (80점 이상 'A', 미만 'B')
grades = np.where(scores >= 80, 'A', 'B')
print("\n등급 부여:")
print(grades)
```

### 7.2 결측치 처리

```python
print("=== 결측치 처리 예제 ===")
# NaN 값이 포함된 데이터
data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
print("원본 데이터:", data)

# NaN 값을 평균으로 대체
mean_val = np.nanmean(data)  # NaN을 제외한 평균
cleaned_data = np.where(np.isnan(data), mean_val, data)
print("결측치 처리 후:", cleaned_data)
```

## 8. 성능 고려사항

### 8.1 메모리 사용량

```python
print("=== 성능 고려사항 ===")
# 큰 배열에서의 연산
large_array = np.random.randn(1000, 1000)

# append는 새로운 배열을 생성 (메모리 사용량 증가)
# 대량 데이터에서는 사전에 크기를 정하고 인덱싱 사용 권장

# 예: 사전 할당된 배열 사용
result_array = np.empty((1100, 1000))  # 미리 크기 할당
result_array[:1000] = large_array      # 데이터 복사
result_array[1000:] = 0                # 나머지 초기화
```

### 8.2 axis 매개변수의 중요성

```python
# axis 매개변수를 명시하여 의도한 연산 수행
def safe_append_row(arr, new_row):
    """안전한 행 추가 함수"""
    if len(new_row.shape) == 1:
        new_row = new_row.reshape(1, -1)
    return np.append(arr, new_row, axis=0)

def safe_append_col(arr, new_col):
    """안전한 열 추가 함수"""
    if len(new_col.shape) == 1:
        new_col = new_col.reshape(-1, 1)
    return np.append(arr, new_col, axis=1)
```

## 9. 정리 및 권장사항

### 핵심 포인트
1. **axis 매개변수 필수**: 2차원 이상 배열에서는 반드시 axis 지정
2. **차원 축소 주의**: axis 미지정 시 예상치 못한 차원 축소 발생
3. **배열 형태 확인**: 추가하려는 데이터의 형태가 기존 배열과 호환되는지 확인
4. **성능 고려**: 큰 데이터에서는 사전 할당 후 인덱싱 사용 권장

### 자주 사용하는 패턴
```python
# 행 추가 패턴
new_data = np.append(existing_array, new_row_data, axis=0)

# 열 추가 패턴  
new_data = np.append(existing_array, new_col_data, axis=1)

# 조건부 처리 패턴
processed_data = np.where(condition, true_values, false_values)

# 안전한 삭제 패턴
cleaned_data = np.delete(array, indices_to_remove, axis=0)
```

이러한 배열 조작 기능들은 데이터 전처리, 특성 엔지니어링, 그리고 머신러닝 파이프라인에서 필수적으로 사용되는 도구들입니다.