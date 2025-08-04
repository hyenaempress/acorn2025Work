# 11. 알고리즘 기초

## 11.1 알고리즘의 개요

### 가. 기본 개념

**알고리즘**이란 문제를 해결하기 위한 단계적 절차 또는 방법을 의미합니다. 즉, 어떤 문제를 해결하기 위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어들을 순서대로 나열한 것입니다.

**기본 구조:** 문제 → 입력 → 알고리즘으로 처리 → 출력

컴퓨터 프로그램을 만들기 위한 알고리즘은 계산과정을 최대한 구체적이고 명료하게 작성해야 합니다.

### 나. 자료구조와 알고리즘의 관계

- **자료구조**란 자료를 배치한 방법을 이야기합니다
- 효율적인 자료구조의 선택 = 효율적인 알고리즘의 선택
- 알고리즘과 프로토콜은 흔히 바꿔서 사용되지만 완전히 같지 않습니다
- **프로토콜은 블록체인의 기본 규칙이며, 알고리즘은 이러한 규칙을 따르는 메커니즘**

## 11.2 기본 알고리즘 예제

### 문제 1: 1부터 10까지의 정수의 합 구하기

```python
import numpy as np

# 방법 1: numpy 사용
numbers = np.arange(1, 11)
total_sum = np.sum(numbers)
print(f"numpy 사용: 1부터 10까지의 정수의 합: {total_sum}")

# 방법 2: numpy 함수로 구현 (O(n) 시간 복잡도)
def totFunc1(n):
    return np.sum(np.arange(1, n + 1))

print(f"totFunc1(10): {totFunc1(10)}")

# 방법 3: 반복문 사용 (O(n) 시간 복잡도)
def totFunc2(n):
    tot = 0
    for i in range(1, n + 1):
        tot += i
    return tot

print(f"totFunc2(10): {totFunc2(10)}")

# 방법 4: 가우스 덧셈 공식 (O(1) 시간 복잡도)
# 이 방법은 가우스 덧셈 공식이라고 불리며, 1부터 n까지의 합을 빠르게 계산할 수 있습니다.
def totFunc3(n):
    return (1 + n) * n // 2

print(f"totFunc3(10): {totFunc3(10)}")
```

### 문제 2: 임의의 정수들 중 최대값/최소값 찾기

```python
# 테스트 데이터
d = [12, 45, 7, 89, 23, 56, 92]
print(f"테스트 데이터: {d}")

# 최대값 찾기 - 방법 1 (for-in 루프)
def findMaxFunc(a):
    max_v = a[0]  # 첫 번째 요소를 최대값으로 초기화
    for i in a:  # 리스트의 각 요소를 순회
        if i > max_v:
            max_v = i
    return max_v  # 최대값 반환

print(f"findMaxFunc: {findMaxFunc(d)}")

# 최대값 찾기 - 방법 2 (인덱스 사용)
def findMaxFunc2(a):
    max_v = a[0]  # 첫 번째 요소를 최대값으로 초기화
    for i in range(1, len(a)):  # 리스트의 각 요소를 순회
        if a[i] > max_v:
            max_v = a[i]
    return max_v  # 최대값 반환

print(f"findMaxFunc2: {findMaxFunc2(d)}")

# 최소값 찾기
def findMinFunc(a):
    min_v = a[0]  # 첫 번째 요소를 최소값으로 초기화
    for i in range(1, len(a)):
        if a[i] < min_v:
            min_v = a[i]
    return min_v  # 최소값 반환

print(f"findMinFunc: {findMinFunc(d)}")

# 최대값의 위치(인덱스) 반환
def findMaxIndexFunc(a):
    max_v = a[0]  # 첫 번째 요소를 최대값으로 초기화
    max_index = 0  # 최대값의 인덱스 초기화
    for i in range(1, len(a)):  # 리스트의 각 요소를 순회
        if a[i] > max_v:
            max_v = a[i]
            max_index = i  # 최대값의 인덱스 업데이트
    return max_index  # 최대값의 인덱스 반환

max_idx = findMaxIndexFunc(d)
print(f"최대값의 인덱스: {max_idx}, 최대값: {d[max_idx]}")

# 최소값의 위치(인덱스) 반환
def findMinIndexFunc(a):
    min_v = a[0]  # 첫 번째 요소를 최소값으로 초기화
    min_index = 0  # 최소값의 인덱스 초기화
    for i in range(1, len(a)):  # 리스트의 각 요소를 순회
        if a[i] < min_v:
            min_v = a[i]
            min_index = i  # 최소값의 인덱스 업데이트
    return min_index  # 최소값의 인덱스 반환

min_idx = findMinIndexFunc(d)
print(f"최소값의 인덱스: {min_idx}, 최소값: {d[min_idx]}")
```

### 문제 3: 동명이인 찾기

```python
# 테스트 데이터
imsi = ['길동', '순신', '길동', '순신', '철수', '영희']
print(f"테스트 데이터: {imsi}")

# 방법 1: set 사용 (중복 제거)
imsi2 = set(imsi)
print(f"set 사용 (중복 제거): {imsi2}")

# 방법 2: 중복된 이름 찾기 (O(n²))
def findSameFunc(a):
    """중복된 이름들을 찾아 반환"""
    n = len(a)
    result = set()
    for i in range(0, n):
        for j in range(i + 1, n):  # i+1부터 시작하여 중복 비교 방지
            if a[i] == a[j]:  # 이름이 같으면?
                result.add(a[i])
    return result

names = ['tom', 'jerry', 'mike', 'tom', 'mike']
print(f"findSameFunc 결과: {findSameFunc(names)}")
print("빅오표기법으로 하면 O(n²)이 된다.")

# 방법 3: 중복 제거하여 고유한 이름들만 반환
def removeDuplicates(names):
    """중복을 제거하여 고유한 이름들만 반환"""
    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names

print(f"함수 사용 (중복 제거): {removeDuplicates(imsi)}")

# 방법 4: 중복된 이름만 찾기 (동명이인만 반환)
def findDuplicates(names):
    """중복된 이름들만 반환"""
    duplicates = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i] == names[j] and names[i] not in duplicates:
                duplicates.append(names[i])
    return duplicates

print(f"중복된 이름만: {findDuplicates(imsi)}")

# 방법 5: 각 이름의 개수 세기
def countNames(names):
    """각 이름의 개수를 딕셔너리로 반환"""
    name_count = {}
    for name in names:
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1
    return name_count

name_counts = countNames(imsi)
print(f"각 이름의 개수: {name_counts}")

# 방법 6: 동명이인만 개수와 함께 출력
def findDuplicatesWithCount(names):
    """중복된 이름과 그 개수를 반환"""
    name_count = countNames(names)
    duplicates = {}
    for name, count in name_count.items():
        if count > 1:
            duplicates[name] = count
    return duplicates

duplicates_with_count = findDuplicatesWithCount(imsi)
print(f"동명이인과 개수: {duplicates_with_count}")

# 방법 7: 효율적인 중복 찾기 (O(n))
def findDuplicatesEfficient(names):
    """효율적으로 중복된 이름들만 반환"""
    seen = set()
    duplicates = set()
    
    for name in names:
        if name in seen:
            duplicates.add(name)
        else:
            seen.add(name)
    
    return list(duplicates)

print(f"효율적인 중복 찾기: {findDuplicatesEfficient(imsi)}")
```

### 문제 4: 팩토리얼과 피보나치 수열 - 재귀 함수 활용

```python
# 팩토리얼 구하기
print("--- 팩토리얼 (5!) ---")

# 방법 1: for문 사용
def factFunc(n):
    """반복문을 사용한 팩토리얼 계산"""
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result

print(f"반복문 사용 factFunc(5): {factFunc(5)}")

# 방법 2: 재귀 호출 사용
def factFunc2(n):
    """재귀 함수를 사용한 팩토리얼 계산"""
    # 재귀의 경우 반드시 빠져 나갈 수 있는 조건이 있어야 한다.
    if n <= 1:  # 종료 조건 필수, 없으면 무한 루프
        return 1
    return n * factFunc2(n - 1)  # 자신이 자기를 부르는 재귀 호출

print(f"재귀 함수 사용 factFunc2(5): {factFunc2(5)}")

# 피보나치 수열
print("--- 피보나치 수열 ---")

# 방법 1: 반복문 사용
def fibFunc(n):
    """반복문을 사용한 피보나치 수열"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for i in range(2, n + 1):
        next_fib = a + b
        a, b = b, next_fib
    return b

print(f"반복문 사용 fibFunc(10): {fibFunc(10)}")

# 방법 2: 재귀 함수 사용
def fibFunc2(n):
    """재귀 함수를 사용한 피보나치 수열"""
    if n <= 1:  # 종료 조건
        return n
    return fibFunc2(n - 1) + fibFunc2(n - 2)  # 재귀 호출

print(f"재귀 함수 사용 fibFunc2(10): {fibFunc2(10)}")

# 피보나치 수열 출력 (처음 10개)
print("피보나치 수열 (처음 10개):")
for i in range(10):
    print(f"F({i}) = {fibFunc(i)}")
```

## 11.3 알고리즘 설계 기법

### 1. 분할정복(Divide & Conquer)
- 문제를 더 이상 나눌 수 없을 때까지 나누고, 나누어진 문제들을 각각 풀어서 전체 문제의 답을 얻는 방법
- **예시:** 병합정렬, 거듭제곱

### 2. 동적계획법(Dynamic Programming)
- 어떤 문제가 여러 단계의 반복되는 부분 문제로 이루어질 때, 각 단계의 부분 문제 답을 기반으로 전체 문제의 답을 구하는 방법
- **예시:** 피보나치 수, LCS 알고리즘

### 3. 탐욕법(Greedy Method)
- 각 단계에서 최선의 선택을 하여 전체적인 최적해를 구하려는 전략
- **주의:** 최적의 해를 구한다는 보장이 없음
- **예시:** 크루스칼, 다익스트라 알고리즘, 허프만 트리

### 4. 퇴각 검색법(Backtracking)
- 여러 후보해 중에서 특정 조건을 충족시키는 모든 해를 찾는 알고리즘
- **목적:** "진짜 해"를 효율적으로 찾는 것
- **예시:** N-Queen 알고리즘

## 11.4 알고리즘의 평가

### 빅오(Big-O) 표기법

빅오 표기법은 알고리즘의 효율성을 표기해주는 표기법입니다. 알고리즘의 효율성은 데이터 개수(n)가 주어졌을 때 기본 연산의 횟수를 의미합니다.

**빅오 표기법의 특징:**
1. **상수항 무시:** 데이터 입력값(n)이 충분히 크다고 가정하므로 상수항은 무시
2. **영향력 없는 항 무시:** 가장 영향력이 큰 항 이외에는 무시

### 시간 복잡도(Time Complexity)
입력크기의 값에 대하여 단위연산을 몇 번 수행하는지를 계산하여 알고리즘의 수행시간을 평가

### 공간 복잡도(Space Complexity)
알고리즘 수행에 필요한 메모리의 양을 평가 (필요한 고정 공간과 가변 공간의 합)

## 11.5 주요 시간 복잡도

| 복잡도 | 설명 | 예시 |
|--------|------|------|
| O(1) | 상수 시간 | 스택에서 Push, Pop |
| O(log n) | 로그 시간 | 이진트리 탐색 |
| O(n) | 선형 시간 | for 문 한 번 |
| O(n log n) | 선형로그 시간 | 퀵 정렬, 병합정렬, 힙 정렬 |
| O(n²) | 이차 시간 | 이중 for 문, 삽입정렬, 거품정렬, 선택정렬 |
| O(2ⁿ) | 지수 시간 | 피보나치 수열(단순 재귀) |

## 11.6 최종 시간 복잡도 분석

### 【기본 알고리즘】
- **totFunc1, totFunc2:** O(n) - 입력 크기에 비례하여 시간 증가
- **totFunc3:** O(1) - 입력 크기와 관계없이 일정한 시간
- **findMaxFunc 계열:** O(n) - 리스트의 모든 요소를 한 번씩 확인

### 【동명이인 찾기】
- **findSameFunc, findDuplicates:** O(n²) - 모든 쌍을 비교
- **removeDuplicates:** O(n²) - 중첩 반복문으로 인한 이차 시간
- **countNames:** O(n) - 한 번의 순회
- **findDuplicatesEfficient:** O(n) - set의 효율적인 탐색 활용

### 【재귀 함수】
- **factFunc (반복문):** O(n) - n번 반복
- **factFunc2 (재귀):** O(n) - n번 재귀 호출
- **fibFunc (반복문):** O(n) - n번 반복
- **fibFunc2 (재귀):** O(2^n) - 지수적 증가 (비효율적)

### 【공간 복잡도】
- **반복문 알고리즘:** 대부분 O(1) - 일정한 메모리
- **재귀 알고리즘:** O(n) - 호출 스택으로 인한 메모리 사용
- **자료구조 사용:** O(n) - 추가 데이터 저장

## 11.7 재귀 함수의 특징

### 재귀 함수 설계 원칙
1. **종료 조건(Base Case):** 재귀 호출을 중단할 조건이 반드시 필요
2. **점진적 접근:** 매번 호출마다 문제가 작아져야 함
3. **자기 참조:** 함수가 자신을 호출하는 구조

### 재귀 vs 반복문 비교

| 구분 | 재귀 함수 | 반복문 |
|------|-----------|--------|
| **가독성** | 직관적, 수학적 정의와 유사 | 명시적 제어 구조 |
| **메모리 사용** | 호출 스택 사용 (O(n)) | 일정한 메모리 (O(1)) |
| **성능** | 함수 호출 오버헤드 | 일반적으로 더 빠름 |
| **디버깅** | 스택 추적 어려움 | 디버깅 용이 |

## 11.8 알고리즘 최적화 전략

### 1. 효율성 우선순위
1. **알고리즘 개선:** 시간 복잡도 자체를 줄이기
2. **자료구조 최적화:** 적절한 자료구조 선택
3. **구현 최적화:** 상수 팩터 개선

### 2. 실무 적용 가이드
- **작은 데이터:** 간단하고 이해하기 쉬운 알고리즘 선택
- **큰 데이터:** 시간 복잡도를 최우선으로 고려
- **메모리 제약:** 공간 복잡도와 시간 복잡도의 트레이드오프 고려
- **유지보수:** 코드의 가독성과 성능의 균형

## 11.9 알고리즘 학습 팁

1. **문제 이해:** 무엇을 해결하려는지 명확히 파악
2. **단계적 접근:** 복잡한 문제를 작은 단위로 분해
3. **효율성 고려:** 시간과 공간 복잡도를 항상 염두에 두기
4. **다양한 방법 시도:** 여러 알고리즘 기법을 적용해보기
5. **실습 중심:** 이론과 함께 직접 구현해보기
6. **성능 측정:** 실제 데이터로 성능 비교하기

효율적인 알고리즘 선택은 프로그램의 성능을 크게 좌우하므로, 문제의 특성을 파악하고 적절한 알고리즘을 선택하는 것이 중요합니다.