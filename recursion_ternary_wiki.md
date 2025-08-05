# 재귀(Recursion)와 삼항 연산자 완전 정리

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

## 4. 재귀의 메모리 구조와 포인터

### 메모리 영역 이해

```
┌─────────────────┐ ← 높은 주소
│      스택       │   - 지역변수, 매개변수
│    (Stack)      │   - 함수 호출 정보
│       ↓         │   - 자동 관리
├─────────────────┤
│       ...       │
├─────────────────┤ 
│       힙        │   - 동적 할당 메모리
│     (Heap)      │   - malloc, new
│       ↑         │   - 수동 관리
├─────────────────┤
│   데이터 영역    │   - 전역변수, 정적변수
├─────────────────┤
│   코드 영역      │   - 실행할 프로그램 코드
└─────────────────┘ ← 낮은 주소
```

### 스택 프레임(Stack Frame) 구조

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

# factorial(3) 호출 시 스택 상태
```

```
스택 메모리 상태:

factorial(3) 호출
┌──────────────────┐ ← SP (Stack Pointer)
│ n = 3           │
│ 리턴 주소        │
│ 이전 프레임 포인터│
├──────────────────┤
│ factorial(2) 호출│
│ n = 2           │ 
│ 리턴 주소        │
│ 이전 프레임 포인터│
├──────────────────┤
│ factorial(1) 호출│
│ n = 1           │
│ 리턴 주소        │
│ 이전 프레임 포인터│
└──────────────────┘

각 프레임 구성요소:
- 매개변수 (n)
- 지역변수
- 리턴 주소 (돌아갈 위치)
- 이전 프레임 포인터
```

### 포인터와 메모리 관리

#### 1) 스택 포인터 (Stack Pointer)
```c
// C언어 예제로 스택 포인터 이해
int recursive_func(int n) {
    int local_var = n;  // 스택에 할당
    printf("주소: %p, 값: %d\n", &local_var, local_var);
    
    if (n <= 1) return 1;
    return n * recursive_func(n-1);  // 새로운 스택 프레임 생성
}
```

#### 2) 프레임 포인터 (Frame Pointer)
```python
def trace_stack(n, depth=0):
    """스택 프레임 추적 함수"""
    import sys
    
    # 현재 프레임 정보
    frame = sys._getframe()
    print(f"{'  ' * depth}프레임 {depth}: n={n}, 주소={id(frame)}")
    
    if n <= 1:
        return 1
    return n * trace_stack(n-1, depth+1)

# 실행 결과:
# 프레임 0: n=3, 주소=140234567890
#   프레임 1: n=2, 주소=140234567891  
#     프레임 2: n=1, 주소=140234567892
```

## 5. 스택 오버플로우 완전 분석

### 1) 스택 오버플로우란?

스택 영역의 메모리가 모두 소진되어 더 이상 함수 호출을 할 수 없는 상태

#### 스택 크기 확인
```python
import sys
print(f"현재 재귀 한계: {sys.getrecursionlimit()}")  # 기본값: 1000
print(f"현재 스택 크기: {sys.getsizeof(sys._getframe())}")

# 한계 조정 (주의!)
sys.setrecursionlimit(2000)  # 2000까지 증가
```

### 2) 스택 오버플로우 실제 테스트

```python
import sys

def test_stack_overflow(n, call_count=0):
    """스택 오버플로우 테스트"""
    try:
        if call_count % 100 == 0:  # 100번마다 출력
            print(f"호출 횟수: {call_count}, 매개변수: {n}")
            
        if n <= 0:
            return "기저 조건 도달"
            
        return test_stack_overflow(n-1, call_count+1)
        
    except RecursionError as e:
        return f"스택 오버플로우 발생! 호출 횟수: {call_count}"

# 테스트 실행
result = test_stack_overflow(2000)
print(result)

# 출력 예시:
# 호출 횟수: 0, 매개변수: 2000
# 호출 횟수: 100, 매개변수: 1900
# 호출 횟수: 200, 매개변수: 1800
# ...
# 스택 오버플로우 발생! 호출 횟수: 996
```

### 3) 스택 오버플로우 원인 분석

#### 잘못된 기저 조건
```python
# 🚫 잘못된 예 1: 기저 조건 없음
def infinite_recursion(n):
    print(f"n = {n}")
    return infinite_recursion(n-1)  # 영원히 감소

# 🚫 잘못된 예 2: 잘못된 기저 조건  
def wrong_base_case(n):
    if n == 5:  # n이 5가 아닌 값으로 시작하면?
        return 1
    return wrong_base_case(n-1)

# 🚫 잘못된 예 3: 값이 증가하는 재귀
def increasing_recursion(n):
    if n <= 0:
        return 1
    return increasing_recursion(n+1)  # 계속 증가!

# ✅ 올바른 예
def correct_recursion(n):
    if n <= 0:      # 명확한 기저 조건
        return 1
    return n * correct_recursion(n-1)  # 값이 감소
```

### 4) 스택 프레임 메모리 사용량 측정

```python
import tracemalloc
import sys

def measure_stack_memory():
    """스택 메모리 사용량 측정"""
    tracemalloc.start()
    
    def deep_recursion(n, depth=0):
        if depth == 0:
            # 초기 메모리 측정
            current, peak = tracemalloc.get_traced_memory()
            print(f"시작 메모리: {current / 1024:.2f} KB")
        
        if n <= 0:
            # 최대 깊이에서 메모리 측정
            current, peak = tracemalloc.get_traced_memory()
            print(f"최대 메모리: {current / 1024:.2f} KB")
            print(f"피크 메모리: {peak / 1024:.2f} KB")
            return depth
            
        return deep_recursion(n-1, depth+1)
    
    result = deep_recursion(500)
    tracemalloc.stop()
    return result

# 실행
measure_stack_memory()
```

### 5) 스택 오버플로우 해결 방법

#### 방법 1: 반복문 변환
```python
# 재귀 버전 (스택 오버플로우 위험)
def factorial_recursive(n):
    if n <= 1:
        return 1
    return n * factorial_recursive(n-1)

# 반복문 버전 (안전)
def factorial_iterative(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
```

#### 방법 2: 꼬리 재귀 최적화 (수동)
```python
# 일반 재귀 (스택 누적)
def sum_recursive(n):
    if n <= 0:
        return 0
    return n + sum_recursive(n-1)

# 꼬리 재귀 스타일 (누적값 전달)
def sum_tail_recursive(n, accumulator=0):
    if n <= 0:
        return accumulator
    return sum_tail_recursive(n-1, accumulator + n)

# 파이썬은 꼬리 재귀 최적화를 지원하지 않으므로
# 트램폴린 패턴 사용
def trampoline(func):
    """트램폴린 패턴으로 스택 오버플로우 방지"""
    while callable(func):
        func = func()
    return func

def sum_trampoline(n, acc=0):
    if n <= 0:
        return acc
    return lambda: sum_trampoline(n-1, acc + n)

# 사용법
result = trampoline(lambda: sum_trampoline(10000))
print(result)  # 50005000 (스택 오버플로우 없이!)
```

#### 방법 3: 스택 크기 증가 (임시 방편)
```python
import sys
import threading

# 방법 1: 재귀 한계 증가
sys.setrecursionlimit(10000)  # 기본 1000에서 10000으로

# 방법 2: 스레드 스택 크기 증가
def run_with_large_stack(func, *args, **kwargs):
    """큰 스택 크기로 함수 실행"""
    def wrapper():
        return func(*args, **kwargs)
    
    # 스택 크기 8MB로 설정
    thread = threading.Thread(target=wrapper, 
                            stack_size=8*1024*1024)
    thread.start()
    thread.join()

# 사용 예
def deep_function(n):
    if n <= 0:
        return "완료"
    return deep_function(n-1)

run_with_large_stack(deep_function, 5000)
```

### 6) 실제 스택 오버플로우 디버깅

```python
import sys
import traceback

def debug_stack_overflow():
    """스택 오버플로우 상황 분석"""
    
    def problematic_function(n, call_stack=[]):
        # 호출 스택 추적
        call_stack.append(n)
        
        try:
            if len(call_stack) > 10:  # 처음 10개만 기록
                call_stack = call_stack[-10:]
                
            if n <= 0:
                return "성공"
            return problematic_function(n-1, call_stack)
            
        except RecursionError:
            print("=== 스택 오버플로우 디버그 정보 ===")
            print(f"최대 재귀 한계: {sys.getrecursionlimit()}")
            print(f"마지막 10개 호출: {call_stack[-10:]}")
            print(f"총 호출 횟수 (추정): {len(call_stack)}")
            
            # 스택 트레이스 출력
            print("\n스택 트레이스:")
            traceback.print_exc()
            
            return "스택 오버플로우 발생"
    
    return problematic_function(2000)

# 실행
debug_stack_overflow()
```

### 7) 메모리 효율적인 재귀 패턴

```python
# 메모이제이션으로 중복 계산 방지
def fibonacci_memo(n, memo={}):
    """메모이제이션 피보나치"""
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
        
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# 제너레이터를 이용한 지연 평가
def fibonacci_generator():
    """무한 피보나치 제너레이터"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 사용법
fib_gen = fibonacci_generator()
for i, fib_num in enumerate(fib_gen):
    if i >= 10:  # 처음 10개만
        break
    print(f"F({i}) = {fib_num}")
```

## 6. 재귀 사용 시 주의사항

### 1) 성능 문제
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

## 7. 언제 재귀를 사용할까?

### 재귀가 좋은 경우
- **트리 구조** 탐색 (파일 시스템, DOM)
- **분할 정복** 알고리즘 (퀵소트, 머지소트)
- **수학적 정의**가 재귀적인 경우 (팩토리얼, 피보나치)
- **백트래킹** 문제 (N-Queen, 미로 찾기)

### 반복문이 좋은 경우
- **단순 반복** 작업
- **메모리 효율성**이 중요한 경우
- **성능**이 중요한 경우

## 8. 실전 연습 문제

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