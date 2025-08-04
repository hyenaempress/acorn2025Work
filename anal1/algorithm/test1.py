import numpy as np

# 알고리즘은 문제를 해결하기 위한 단계적 절차 또는 방법을 의미한다
# 즉 어떤 문제를 해결하기위해 컴퓨터가 따라 할 수 있도록 구체적인 명령어들을 순서대로 나열한것
# 컴퓨터 프로그램을 만들기 위한 알고리즘은 계산과정을 최대한 구체적이고 명료하게 작성해야 합니다.
# 문제 -> 입력 -> 알고리즘으로 처리 -> 출력

print("=" * 60)
print("문제 1: 1부터 10까지의 정수의 합 구하기")
print("=" * 60)

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

print("\n" + "=" * 60)
print("문제 2: 임의의 정수들 중 최대값/최소값 찾기")
print("=" * 60)

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

print("\n" + "=" * 60)
print("문제 3: 동명이인 찾기 - n명의 사람 이름 중 동일한 이름을 찾아 결과 출력")
print("=" * 60)

imsi = ['길동', '순신', '길동', '순신', '철수', '영희']
print(f"테스트 데이터: {imsi}")

# 방법 1: set 사용 (중복 제거)
imsi2 = set(imsi)
print(f"\nset 사용 (중복 제거): {imsi2}")

# 방법 2: 중복된 이름 찾기 (수정된 버전)
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

# 방법 3: set을 사용하지 않고 중복 제거
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
print(f"\n각 이름의 개수: {name_counts}")

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

# 방법 7: 좀 더 효율적인 중복 찾기 (한 번의 순회로)
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

print("\n" + "=" * 60)
print("문제 4: 팩토리얼과 피보나치 수열 - 재귀 함수 활용")
print("=" * 60)

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

# 방법 2: 재귀 호출 사용 (수정된 버전)
def factFunc2(n):
    """재귀 함수를 사용한 팩토리얼 계산"""
    # 재귀의 경우 반드시 빠져 나갈 수 있는 조건이 있어야 한다.
    if n <= 1:  # 종료 조건 필수, 없으면 무한 루프
        return 1
    return n * factFunc2(n - 1)  # 자신이 자기를 부르는 재귀 호출

print(f"재귀 함수 사용 factFunc2(5): {factFunc2(5)}")

# 피보나치 수열
print("\n--- 피보나치 수열 ---")

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

#재귀 연습 1)1부터 n 까지의 합 구하기 : 재귀 사용
def hap(n):
    if n <= 1:  # 종료 조건
        return n
    return n + hap(n-1)     

#재귀 연습 2)숫자 n 중 최대값 구하기 : 재귀 사용 
def findMax(a,n):
    if n ==1:
        return a[0]
    max_size_find = findMax(a, n-1) #비교 후 카운트 내리기 카운트 1일떄 계산 
    return max(a[n-1] ,max_size_find)
valuse = [7,9,15,42,33,22]
print(findMax(valuse, len(valuse)))

def findMax2(a, n):
    # 종료 조건: 원소가 1개만 남았을 때
    if n == 1:
        return a[0]
    
    # 재귀: n-1개의 최대값 구하기
    max_of_rest = findMax2(a, n-1)
    current = a[n-1]  # 현재값
    
    # 직접 비교해서 큰 값 리턴
    if current > max_of_rest:
        return current
    else:
        return max_of_rest

values = [7, 9, 15, 42, 33, 22]
print(findMax2(values, len(values)))  # 출력: 42

def findMax3(a, n):
    if n == 1:
        return a[0]
    
    max_of_rest = findMax3(a, n-1)
    # 삼항 연산자로 비교
    return a[n-1] if a[n-1] > max_of_rest else max_of_rest
values = [7, 9, 15, 42, 33, 22]
print(findMax3(values, len(values)))  # 출력: 42
# 조건문을 한 줄로 간단하게 표현하는 연산자를 삼항 연산자라고 한다 


print("\n" + "=" * 60)
print("최종 시간 복잡도 분석")
print("=" * 60)
print("【기본 알고리즘】")
print("• totFunc1, totFunc2: O(n) - 입력 크기에 비례하여 시간 증가")
print("• totFunc3: O(1) - 입력 크기와 관계없이 일정한 시간")
print("• findMaxFunc 계열: O(n) - 리스트의 모든 요소를 한 번씩 확인")
print("\n【동명이인 찾기】")
print("• findSameFunc, findDuplicates: O(n²) - 모든 쌍을 비교")
print("• removeDuplicates: O(n²) - 중첩 반복문으로 인한 이차 시간")
print("• countNames: O(n) - 한 번의 순회")
print("• findDuplicatesEfficient: O(n) - set의 효율적인 탐색 활용")
print("\n【재귀 함수】")
print("• factFunc (반복문): O(n) - n번 반복")
print("• factFunc2 (재귀): O(n) - n번 재귀 호출")
print("• fibFunc (반복문): O(n) - n번 반복")
print("• fibFunc2 (재귀): O(2^n) - 지수적 증가 (비효율적)")
print("\n【공간 복잡도】")
print("• 반복문 알고리즘: 대부분 O(1) - 일정한 메모리")
print("• 재귀 알고리즘: O(n) - 호출 스택으로 인한 메모리 사용")
print("• 자료구조 사용: O(n) - 추가 데이터 저장")