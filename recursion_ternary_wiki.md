# 재귀(Recursion)와 삼항 연산자 정리

## 1. 재귀(Recursion)란?

### 정의
**재귀**란 함수가 자기 자신을 호출하는 프로그래밍 기법입니다.

### 재귀의 핵심 구성 요소

#### 1) 기저 조건(Base Case)
- 재귀 호출을 멈추는 조건
- 없으면 **무한 재귀**로 스택 오버플로우 발생
- 가장 간단한 경우의 해답

#### 2) 재귀 호출(Recursive Call)  
- 자기 자신을 더 작은 문제로 호출
- 문제를 점점 작게 만들어 기저 조건에 도달

### 재귀 작동 원리

```python
def countdown(n):
    if n <= 0:        # 기저 조건
        print("발사!")
        return
    print(n)
    countdown(n-1)    # 재귀 호출 (더 작은 문제)

# 실행 과정
countdown(3)
# 출력:
# 3
# 2  
# 1
# 발사!
```

### 재귀 vs 반복문

| 구분 | 재귀 | 반복문 |
|------|------|--------|
| **가독성** | 직관적, 수학적 | 명시적 |
| **메모리** | 스택 사용 (많음) | 변수 사용 (적음) |
| **성능** | 함수 호출 오버헤드 | 빠름 |
| **적용** | 트리, 그래프, 수학 문제 | 단순 반복 |

## 2. 재귀 연습 예제

### 예제 1: 1부터 n까지의 합

```python
def hap(n):
    if n <= 1:          # 기저 조건
        return n
    return n + hap(n-1) # 재귀: n + (1부터 n-1까지의 합)

# 작동 과정
# hap(5) = 5 + hap(4)
# hap(4) = 4 + hap(3)  
# hap(3) = 3 + hap(2)
# hap(2) = 2 + hap(1)
# hap(1) = 1 (기저 조건)
# 결과: 5 + 4 + 3 + 2 + 1 = 15
```

### 예제 2: 배열에서 최대값 찾기

```python
def findMax(a, n):
    if n == 1:                    # 기저 조건: 원소 1개
        return a[0]
    
    max_of_rest = findMax(a, n-1) # 재귀: 나머지 원소들의 최대값
    current = a[n-1]              # 현재 원소
    
    # 비교해서 큰 값 반환
    if current > max_of_rest:
        return current
    else:
        return max_of_rest

values = [7, 9, 15, 42, 33, 22]
print(findMax(values, len(values)))  # 42
```

### 호출 스택 시각화

```
findMax([7,9,15,42,33,22], 6)
├─ findMax([7,9,15,42,33,22], 5) 
   ├─ findMax([7,9,15,42,33,22], 4)
      ├─ findMax([7,9,15,42,33,22], 3)
         ├─ findMax([7,9,15,42,33,22], 2)
            ├─ findMax([7,9,15,42,33,22], 1)
            └─ return 7 (기저 조건)
         └─ max(9, 7) = 9
      └─ max(15, 9) = 15  
   └─ max(42, 15) = 42
└─ max(33, 42) = 42
최종 결과: max(22, 42) = 42
```

## 3. 삼항 연산자(Ternary Operator)

### 정의
**삼항 연산자**는 조건문을 한 줄로 간단하게 표현하는 연산자입니다.

### 문법
```python
# 기본 형태
결과 = 값1 if 조건 else 값2

# 조건이 True면 값1, False면 값2를 반환
```

### 일반 if문 vs 삼항 연산자

```python
# 일반 if문
def findMax_normal(a, n):
    if n == 1:
        return a[0]
    
    max_of_rest = findMax_normal(a, n-1)
    
    if a[n-1] > max_of_rest:  # 4줄
        return a[n-1]
    else:
        return max_of_rest

# 삼항 연산자 사용
def findMax_ternary(a, n):
    if n == 1:
        return a[0]
    
    max_of_rest = findMax_ternary(a, n-1)
    return a[n-1] if a[n-1] > max_of_rest else max_of_rest  # 1줄
```

### 삼항 연산자 예제들

```python
# 1. 절댓값 구하기
def abs_value(x):
    return x if x >= 0 else -x

# 2. 최대값 구하기  
def max_two(a, b):
    return a if a > b else b

# 3. 짝수/홀수 판별
def even_odd(n):
    return "짝수" if n % 2 == 0 else "홀수"

# 4. 성인/미성년자 판별
def age_check(age):
    return "성인" if age >= 18 else "미성년자"

# 5. 중첩 삼항 연산자 (권장하지 않음)
def grade(score):
    return "A" if score >= 90 else "B" if score >= 80 else "C"
```

## 4. 재귀 사용 시 주의사항

### 1) 스택 오버플로우
```python
# 잘못된 예: 기저 조건 없음
def bad_recursion(n):
    return bad_recursion(n-1)  # 무한 재귀!

# 올바른 예
def good_recursion(n):
    if n <= 0:      # 기저 조건 필수!
        return 0
    return good_recursion(n-1)
```

### 2) 성능 문제
```python
# 비효율적인 피보나치 (지수 시간복잡도)
def fib_slow(n):
    if n <= 1:
        return n
    return fib_slow(n-1) + fib_slow(n-2)  # 중복 계산 많음

# 효율적인 방법 (메모이제이션)
def fib_fast(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_fast(n-1, memo) + fib_fast(n-2, memo)
    return memo[n]
```

## 5. 언제 재귀를 사용할까?

### 재귀가 좋은 경우
- **트리 구조** 탐색 (파일 시스템, DOM)
- **분할 정복** 알고리즘 (퀵소트, 머지소트)
- **수학적 정의**가 재귀적인 경우 (팩토리얼, 피보나치)
- **백트래킹** 문제 (N-Queen, 미로 찾기)

### 반복문이 좋은 경우
- **단순 반복** 작업
- **메모리 효율성**이 중요한 경우
- **성능**이 중요한 경우

## 6. 실전 연습 문제

```python
# 1. 팩토리얼
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# 2. 거듭제곱
def power(base, exp):
    if exp == 0:
        return 1
    return base * power(base, exp-1)

# 3. 배열 뒤집기
def reverse_array(arr, start, end):
    if start >= end:
        return
    arr[start], arr[end] = arr[end], arr[start]
    reverse_array(arr, start+1, end-1)

# 4. 이진 탐색
def binary_search(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid-1)
    else:
        return binary_search(arr, target, mid+1, right)
```

## 요약

### 재귀 핵심
1. **기저 조건** 반드시 설정
2. **문제를 작게** 만들어 호출
3. **직관적이지만** 메모리 사용량 주의

### 삼항 연산자 핵심  
1. **조건 ? 참값 : 거짓값** 형태
2. **간단한 조건문**을 한 줄로
3. **가독성** 고려해서 사용

재귀는 **"자기가 자기를 호출하되, 점점 작은 문제로!"**  
삼항 연산자는 **"조건에 따라 값을 선택!"**