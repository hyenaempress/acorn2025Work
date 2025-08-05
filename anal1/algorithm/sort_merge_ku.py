


import random
import time

# 정렬 알고리즘의 시간을 측정하는 함수
def measure_sort_time(sort_function, data, label=None, show_result=True):
    start_time = time.time()  # 시작 시간 기록
    sorted_data = sort_function(data)  # 정렬 함수 실행
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 경과 시간 계산
    if label is not None:
        print(f"\n{label} 실행 시간: {elapsed_time:.6f}초")
        if show_result:
            print(f"정렬된 리스트: {sorted_data[:100]}...")  # 정렬된 리스트의 처음 100개 요소만 출력
    return sorted_data, elapsed_time
# ---------- 정렬 알고리즘 구현부 ----------

# 선택 정렬(Selection Sort)은 주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 최소값 찾기: 정렬되지 않은 부분에서 가장 작은 값을 찾는다
# 2. 교환: 찾은 최소값을 정렬되지 않은 부분의 첫 번째 위치와 교환한다
# 3. 반복: 정렬된 부분을 확장하며 위 과정을 반복한다

# 선택 정렬 방법 1: 새로운 리스트(result)를 만들어 정렬 (메모리 사용 많음)
# 시간 복잡도: O(n^2)
# 공간 복잡도: O(n) (result 리스트 사용)
def find_min_value(list_arguement): # 최소값 찾기 함수
    length = len(list_arguement)
    index = 0
    for i in range(1, length):
        if list_arguement[i] < list_arguement[index]:
            index = i
    return list_arguement[index]

def selection_sort1(list_arguement, verbose=False): # 선택 정렬 함수
    result = []
    while list_arguement:  # 리스트가 비어있지 않을 때까지 반복
        if verbose:
            print("\n현재 리스트 상태:", list_arguement)
        min_value = find_min_value(list_arguement)  # 최소값 찾기
        if verbose:
            print(f"찾은 최소값: {min_value}")
        result.append(min_value)  # 최소값을 결과 리스트에 추가
        if verbose:
            print(f"탐색한 최소 값을 결과 리스트에 추가: {result}")
        list_arguement.remove(min_value)  # 원래 리스트에서 최소값 제거
        if verbose:
            print(f"원래 리스트에서 최소값 제거 후: {list_arguement}")

    if verbose:
        print(f"\n최종 정렬 결과: {result}")
    return result

# 선택 정렬 방법 2: 제자리 정렬 (메모리 절약, 실제 문제 풀이에 적합)
# 시간 복잡도: O(n^2)
# 공간 복잡도: O(1) (제자리 정렬, 추가 메모리 사용 없음)
def selection_sort2(list_arguement, verbose=False): # 제자리 정렬 함수
    if verbose:
        print("선택 정렬(메모리 공간 절약):")
    length = len(list_arguement)
    if verbose:
        print(f"확인한 배열 길이: {length}")

    for i in range(length - 1):
        min_index = i
        for j in range(i + 1, length):
            if verbose:
                print(f"{list_arguement[j]}와 {list_arguement[min_index]}를 비교")
            if list_arguement[j] < list_arguement[min_index]:
                if verbose:
                    print(f"{list_arguement[j]}가 {list_arguement[min_index]}보다 작다")
                min_index = j
                if verbose:
                    print(f"\n갱신된 최소 값[{min_index}]: {list_arguement[min_index]}")
            else:
                if verbose:
                    print(f"{list_arguement[j]}는 {list_arguement[min_index]}보다 크거나 같다")
        if min_index != i:
            if verbose:
                print(f"  교환: list_arguement[{i}]({list_arguement[i]}) <-> list_arguement[{min_index}]({list_arguement[min_index]})")
            list_arguement[i], list_arguement[min_index] = list_arguement[min_index], list_arguement[i]
        else:
            if verbose:
                print("  교환 없음 (이미 최소값)")
        if verbose:
            print(f"  결과: {list_arguement}")
    if verbose:
        print(f"\n최종 정렬 결과: {list_arguement}")
    return list_arguement

# ---------- 선택 정렬 함수 끝 ----------

# 삽입 정렬(Insertion Sort)은 주어진 데이터 리스트를 순차적으로 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 첫 번째 원소는 정렬된 상태로 간주한다
# 2. 두 번째 원소부터 시작하여, 현재 원소를 정렬된 부분에 적절한 위치에 삽입한다
# 3. 이 과정을 반복하여 전체 리스트를 정렬한다

# 삽입 정렬 방법 1: 새로운 리스트(result)를 만들어 정렬 (메모리 사용 많음)
# 시간 복잡도: O(n^2)
# 공간 복잡도: O(n) (result 리스트 사용)
def find_insert_index(list_arguement, value, verbose=False): # 삽입할 위치 찾기 함수
    for i in range(len(list_arguement)):
        if verbose:
            print(f"  [find_insert_index] {value} < {list_arguement[i]} ? ", end="")
        if value < list_arguement[i]:
            if verbose:
                print("-> True, 삽입 위치:", i)
            return i
        if verbose:
            print("-> False")
    if verbose:
        print(f"  [find_insert_index] {value}가 가장 크므로 끝에 삽입 (index {len(list_arguement)})")
    return len(list_arguement)

def insertion_sort1(list_arguement, verbose=False): # 삽입 정렬 함수
    result = []
    step = 1
    if verbose:
        print("최초 전달 받은 리스트:", list_arguement)
    while list_arguement:
        value = list_arguement.pop(0)
        if verbose:
            print(f"\n[{step}단계] 남은 입력 리스트에서 꺼낸 값: {value}")
            print(f"  현재 정렬된 리스트: {result}")
        insert_idx = find_insert_index(result, value, verbose)
        result.insert(insert_idx, value)
        if verbose:
            print(f"  삽입 후 리스트: {result}")
            print(f"  남은 입력 리스트: {list_arguement}")
        step += 1
    if verbose:
        print(f"\n최종 정렬 결과: {result}")
    return result

# 삽입 정렬 방법 2: 제자리 정렬 (실제 문제 풀이에 적합)
# 시간 복잡도: O(n^2)
# 공간 복잡도: O(1) (제자리 정렬, 추가 메모리 사용 없음)
def insertion_sort2(list_arguement, verbose=False): # 삽입 정렬 함수
    length = len(list_arguement)
    for i in range(1, length):
        key = list_arguement[i]
        j = i - 1
        while j >= 0 and list_arguement[j] > key:
            list_arguement[j + 1] = list_arguement[j]
            j -= 1
        list_arguement[j + 1] = key
    if verbose:
        print(f"\n최종 정렬 결과: {list_arguement}")
    return list_arguement

# ---------- 삽입 정렬 함수 끝 ----------

# 합병 정렬(Merge Sort)은 주어진 데이터 리스트를 분할하고 정복하여 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 주어진 리스트를 반으로 나눈다
# 2. 각 부분 리스트를 재귀적으로 합병 정렬한다
# 3. 정렬된 두 부분 리스트를 하나로 합병하여 최종 정렬된 리스트를 만든다

# 합병 정렬 방법 1: 새로운 리스트(result)를 만들어 정렬 (메모리 사용 많음)
# 시간 복잡도: O(n log n)
# 공간 복잡도: O(n)
def merge_sort1(list_arguement, verbose=False):
    # -------------------- 리스트를 반으로 나누는 부분(1개만 남을 때까지 쪼개고 쪼갬)------------------
    if len(list_arguement) <= 1: # 리스트가 1개 이하이면 이미 정렬된 상태
        return list_arguement

    mid = len(list_arguement) // 2 # 리스트를 반으로 나누는 인덱스
    # 리스트가 1인 경우 상기 코드에서 return 되므로 mid는 항상 1 이상의 값을 가짐
    if verbose:
        print(f"분할: {list_arguement[:mid]} | {list_arguement[mid:]}") # 분할된 리스트 출력

    left_half = merge_sort1(list_arguement[:mid], verbose) # 왼쪽 절반 정렬
    right_half = merge_sort1(list_arguement[mid:], verbose) # 오른쪽 절반 정렬

    # -------------------- 리스트를 반으로 나누는 부분(1개만 남을 때까지 쪼개고 쪼갬)------------------

    # -------------------- 실제 데이터가 정렬되는 부분(쪼개진 뭉탱이를 비교하여 합침)--------------------
    merged_list = [] # 합병 결과를 저장할 리스트
    i = j = 0 # 왼쪽과 오른쪽 절반의 인덱스 초기화

    while i < len(left_half) and j < len(right_half): # 왼쪽과 오른쪽 절반을 비교하여 합병
        if left_half[i] < right_half[j]: # 왼쪽 절반의 값이 더 작으면
            merged_list.append(left_half[i]) # 왼쪽 절반의 값을 추가
            i += 1 # 인덱스 증가
        else: # 오른쪽 절반의 값이 더 작거나 같으면
            merged_list.append(right_half[j]) # 오른쪽 절반의 값을 추가
            j += 1 # 인덱스 증가

    merged_list.extend(left_half[i:]) # 왼쪽 절반에 남은 값들을 추가
    merged_list.extend(right_half[j:]) # 오른쪽 절반에 남은 값들을 추가

    if verbose:
        print(f"합병 결과: {merged_list}") # 합병된 리스트 출력

    return merged_list # 최종 정렬된 리스트 반환
    # -------------------- 실제 데이터가 정렬되는 부분(쪼개진 뭉탱이를 비교하여 합침)--------------------

# 합병 정렬 방법 2: 제자리 정렬 (실제 문제 풀이에 적합)
# 시간 복잡도: O(n log n)
# 공간 복잡도: O(1)
def merge_sort2(list_arguement, verbose=False):
    pass

# -------------------------------------------------------------- 알고리즘 실행 부분 ---------------------------------------------------------------
# 테스트

test_list = random.sample(range(1, 100000), 50000)  # 1부터 100000 사이의 숫자 중에서 50000개를 랜덤하게 선택하여 리스트 생성

print("생성한 테스트 리스트:", test_list)

# 공간 복잡도가 낮을 수록 대규모 데이터 처리에 유리
# 시간 복잡도가 낮을 수록 빠른 정렬이 가능

# 선택 정렬 1번 실행 (O(n^2) 시간, O(n) 공간)
# - 새로운 리스트(result)에 최소값을 하나씩 추가하고, 원본 리스트에서 remove로 값을 제거함
# - remove 연산이 O(n)이라 실제로는 오버헤드가 큼, 원본 리스트는 소모됨(파괴적)
sorted_list, exec_time = measure_sort_time(
    lambda x: selection_sort1(x, verbose=False),
    test_list.copy(),
    label="선택 정렬 1번",
    show_result=False  # 결과 출력 여부 옵션
)

# 선택 정렬 2번 실행 (O(n^2) 시간, O(1) 공간)
# - 리스트 내부에서 인덱스끼리 값을 교환(제자리 정렬, in-place)
# - 추가 메모리 사용 없음, 원본 리스트가 직접 변경됨
# sorted_list, exec_time = measure_sort_time(
#     lambda x: selection_sort2(x, verbose=False),
#     test_list.copy(),
#     label="선택 정렬 2번",
#     show_result=False
# )

# 삽입 정렬 1번 실행 (O(n^2) 시간, O(n) 공간)
# - 새로운 리스트(result)에 값을 삽입할 위치를 찾아 insert로 삽입
# - insert 연산이 O(n)이지만, 파이썬 내부적으로 최적화되어 있을 수 있음
# - 원본 리스트는 소모됨(파괴적)
sorted_list, exec_time = measure_sort_time(
    lambda x: insertion_sort1(x, verbose=False),
    test_list.copy(),
    label="삽입 정렬 1번",
    show_result=False
)

# 삽입 정렬 2번 실행 (O(n^2) 시간, O(1) 공간)
# - 리스트 내부에서 값을 이동시키며 정렬(제자리 정렬, in-place)
# - 추가 메모리 사용 없음, 원본 리스트가 직접 변경됨
# sorted_list, exec_time = measure_sort_time(
#     lambda x: insertion_sort2(x, verbose=False),
#     test_list.copy(),
#     label="삽입 정렬 2번",
#     show_result=False
# )

# 합병 정렬 1번 실행
sorted_list, exec_time = measure_sort_time(
    lambda x: merge_sort1(x, verbose=False),
    test_list.copy(),
    label="합병 정렬 1번",
    show_result=False
)

# 합병 정렬 2번 실행
sorted_list, exec_time = measure_sort_time(
    lambda x: merge_sort2(x, verbose=False),
    test_list.copy(),
    label="합병 정렬 2번",
    show_result=False
)

# 합병 정렬 방법 2: 제자리 정렬 (실제 문제 풀이에 적합)
# 시간 복잡도: O(n log n)
# 공간 복잡도: O(1)
def merge_sort2(list_arguement, verbose=False):
    length = len(list_arguement)
    if length <= 1:  # 리스트가 1개 이하이면 이미 정렬된 상태
        return
    mid = length // 2  # 리스트를 반으로 나누는 인덱스
    left_half = list_arguement[:mid]  # 왼쪽 절반
    right_half = list_arguement[mid:]  # 오른쪽 절반
    
    merge_sort2(left_half, verbose)  # 왼쪽 절반 정렬(계속 반으로 나누다가 1이 되면)
    merge_sort2(right_half, verbose)  # 오른쪽 절반 정렬(계속 반으로 나누다가 1이 되면)

    # 두 그룹을 하나로 합치기
    i = j = 0 # 왼쪽, 오른쪽 절반의 인덱스
    ia = 0 # 전체 리스트의 인덱스
    # 왼쪽과 오른쪽 절반을 비교하여 합병
    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]: # 두 리스트의 앞쪽 값들을 비교
            list_arguement[ia] = left_half[i]  # 왼쪽 절반의 값이 더 작으면
            i += 1  # 왼쪽 절반의 인덱스 증가
            ia += 1  # 전체 리스트의 인덱스 증가
        else:  # 오른쪽 절반의 값이 더 작거나 같으면
            list_arguement[ia] = right_half[j]  # 오른쪽 절반의 값을 추가
            j += 1  # 오른쪽 절반의 인덱스 증가
            ia += 1  # 전체 리스트의 인덱스 증가
        # 아직 남아있는 자료들을 추가
    while i < len(left_half):  # 왼쪽 절반에 남은 값들을 추가
        list_arguement[ia] = left_half[i]
        i += 1
        ia += 1
    while j < len(right_half):  # 오른쪽 절반에 남은 값들을 추가
        list_arguement[ia] = right_half[j]
        j += 1
        ia += 1
    if verbose:
        print(f"합병 결과: {list_arguement}")
    return list_arguement  # 최종 정렬된 리스트 반환

def merge_sort3(list_arguement, verbose=False):
    length = len(list_arguement)
    if length <= 1:  # 리스트가 1개 이하이면 이미 정렬된 상태
        return
    mid = length // 2  # 리스트를 반으로 나누는 인덱스
    left_half = list_arguement[:mid]  # 왼쪽 절반
    right_half = list_arguement[mid:]  # 오른쪽 절반
    i = j = 0  # 왼쪽, 오른쪽 절반의 인덱스
    result = []  # 병합 결과를 저장할 리스트

    # 병합
    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            result.append(left_half[i])  # 왼쪽 절반의 값이 더 작으면
            i += 1  # 왼쪽 절반의 인덱스 증가
        else:
            result.append(right_half[j])
            j += 1  # 오른쪽 절반의 인덱스 증가

    # 남은 요소들을 결과에 추가
    result.extend(left_half[i:])
    result.extend(right_half[j:])
    # 정렬된 결과를 원본 리스트에 반영
    for k in range(len(result)):
        list_arguement[k] = result[k]

    if verbose:
        print(f"합병 결과: {list_arguement}")

    return list_arguement  # 최종 정렬된 리스트 반환