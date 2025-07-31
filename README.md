# acorn2025Work
acorn2025 현대로템 선도기업 아카데미(데이터분석)

7/31 표준편차 구하기 [text](anal1/numpy1.py)

넘파이, 강력한 N차월 배열을 가지고 있음. 

# 파이썬이 느린 이유와 NumPy의 해결책

## 파이썬 리스트의 한계

### 1. 메모리 구조의 차이
**Python List (동적 배열)**
- 각 요소가 **PyObject**로 래핑됨
- 실제 데이터와 함께 타입 정보, 참조 카운트 등 메타데이터 저장
- 각 요소가 메모리의 다른 위치에 흩어져 있음 (포인터로 연결)
- 메모리 오버헤드 발생

**NumPy Array (연속 메모리)**
- 순수한 데이터만 연속된 메모리 공간에 저장
- 동일한 데이터 타입으로 통일
- 메모리 접근이 효율적 (캐시 친화적)

### 2. 연산 처리 방식의 차이

#### Python List 연산 (느림)
```python
# Python List로 벡터 덧셈
a = [1, 2, 3, 4, 5]
b = [6, 7, 8, 9, 10]

# 반복문으로 하나씩 처리 (인터프리터 방식)
result = []
for i in range(len(a)):
    result.append(a[i] + b[i])  # 각각이 PyObject 연산
```

#### NumPy Array 연산 (빠름)
```python
import numpy as np

# NumPy Array로 벡터 덧셈
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

# 벡터화 연산 (C 언어 수준에서 처리)
result = a + b  # 전체 배열을 한 번에 처리
```

## NumPy가 빠른 이유

### 1. **벡터화 (Vectorization)**
- 반복문 없이 전체 배열을 한 번에 처리
- C/Fortran으로 구현된 최적화된 함수 사용
- CPU의 SIMD(Single Instruction, Multiple Data) 활용

### 2. **연속 메모리 레이아웃**
- 데이터가 메모리에 연속적으로 배치
- CPU 캐시 효율성 극대화
- 메모리 접근 시간 단축

### 3. **타입 동질성**
- 모든 요소가 같은 데이터 타입
- 타입 체크 오버헤드 제거
- 컴파일러 최적화 가능

### 4. **네이티브 코드 실행**
- 핵심 연산이 C 언어로 구현
- Python 인터프리터 오버헤드 우회
- 직접적인 하드웨어 접근

## 성능 비교 예시

```python
import time
import numpy as np

# 큰 배열 생성
size = 1000000
python_list_a = list(range(size))
python_list_b = list(range(size, size*2))

numpy_array_a = np.arange(size)
numpy_array_b = np.arange(size, size*2)

# Python List 연산 시간 측정
start = time.time()
result_list = [a + b for a, b in zip(python_list_a, python_list_b)]
python_time = time.time() - start

# NumPy Array 연산 시간 측정
start = time.time()
result_numpy = numpy_array_a + numpy_array_b
numpy_time = time.time() - start

print(f"Python List: {python_time:.4f}초")
print(f"NumPy Array: {numpy_time:.4f}초")
print(f"속도 향상: {python_time/numpy_time:.1f}배")
```

## 결론

**파이썬이 느린 이유:**
- 인터프리터 언어 (컴파일 언어 대비)
- 동적 타이핑으로 인한 런타임 오버헤드
- PyObject 래핑으로 인한 메모리 오버헤드

**NumPy의 해결책:**
- C 언어 수준의 성능
- 벡터화 연산으로 반복문 제거
- 효율적인 메모리 사용
- 수치 계산에 특화된 최적화

따라서 대용량 데이터나 수치 계산이 필요한 작업에서는 Python 리스트 대신 **NumPy 배열을 사용하는 것이 필수적**입니다.