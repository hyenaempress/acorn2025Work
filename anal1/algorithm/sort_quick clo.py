# 퀵소트 (Quick Sort)
# 분할 정복 알고리즘을 사용하여 평균 O(n log n)의 성능을 가짐

# 방법 1: 원리 이해를 우선시 (새로운 리스트 생성 방식)
def quick_sort_simple(arr):
    """
    원리 이해를 위한 퀵소트 구현
    - 새로운 리스트를 생성하여 분할
    - 메모리를 더 사용하지만 이해하기 쉬움
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # 중간값을 피벗으로 선택
    left = [x for x in arr if x < pivot]      # 피벗보다 작은 값들
    middle = [x for x in arr if x == pivot]   # 피벗과 같은 값들
    right = [x for x in arr if x > pivot]     # 피벗보다 큰 값들
    
    return quick_sort_simple(left) + middle + quick_sort_simple(right)


# 방법 2: 실무에서 사용하는 in-place 퀵소트
def quick_sort_inplace(arr, low=0, high=None):
    """
    실무에서 사용하는 퀵소트 구현
    - 원본 배열을 직접 수정 (in-place)
    - 메모리 효율적
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 파티션 과정: 피벗을 올바른 위치에 배치
        pivot_index = partition(arr, low, high)
        
        # 피벗 기준으로 왼쪽과 오른쪽을 재귀적으로 정렬
        quick_sort_inplace(arr, low, pivot_index - 1)
        quick_sort_inplace(arr, pivot_index + 1, high)
    
    return arr


def partition(arr, low, high):
    """
    파티션 함수: 피벗을 기준으로 배열을 분할
    - 피벗보다 작은 값들을 왼쪽으로
    - 피벗보다 큰 값들을 오른쪽으로
    """
    pivot = arr[high]  # 마지막 원소를 피벗으로 선택
    i = low - 1        # 작은 원소들의 인덱스
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 스왑
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]  # 피벗을 올바른 위치에
    return i + 1


# 방법 3: 최적화된 버전 (랜덤 피벗 선택)
import random

def quick_sort_optimized(arr, low=0, high=None):
    """
    최적화된 퀵소트
    - 랜덤 피벗 선택으로 최악의 경우 방지
    - 작은 배열에 대해서는 삽입 정렬 사용
    """
    if high is None:
        high = len(arr) - 1
    
    # 작은 배열은 삽입 정렬이 더 효율적
    if high - low + 1 < 10:
        insertion_sort_range(arr, low, high)
        return arr
    
    if low < high:
        # 랜덤 피벗 선택
        random_index = random.randint(low, high)
        arr[random_index], arr[high] = arr[high], arr[random_index]
        
        pivot_index = partition(arr, low, high)
        quick_sort_optimized(arr, low, pivot_index - 1)
        quick_sort_optimized(arr, pivot_index + 1, high)
    
    return arr


def insertion_sort_range(arr, low, high):
    """지정된 범위에서 삽입 정렬 수행"""
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터
    test_data = [2, 4, 5, 1, 3, 8, 7, 6, 9]
    
    print("원본 데이터:", test_data)
    print()
    
    # 방법 1: 원리 이해 버전
    result1 = quick_sort_simple(test_data.copy())
    print("방법 1 (원리 이해):", result1)
    
    # 방법 2: 실무 버전
    data2 = test_data.copy()
    quick_sort_inplace(data2)
    print("방법 2 (실무 버전):", data2)
    
    # 방법 3: 최적화 버전
    data3 = test_data.copy()
    quick_sort_optimized(data3)
    print("방법 3 (최적화):", data3)
    
    print()
    print("시간복잡도:")
    print("- 평균/최선: O(n log n)")
    print("- 최악: O(n²) - 이미 정렬된 배열에서 첫/마지막 원소를 피벗으로 선택할 때")
    print("- 공간복잡도: O(log n) - 재귀 호출 스택")
    
    # 큰 데이터로 성능 테스트
    print("\n=== 성능 비교 ===")
    import time
    
    # 큰 랜덤 데이터 생성
    large_data = [random.randint(1, 1000) for _ in range(1000)]
    
    # 각 알고리즘 실행 시간 측정
    algorithms = [
        ("원리 이해 버전", lambda x: quick_sort_simple(x)),
        ("실무 버전", lambda x: quick_sort_inplace(x.copy())),
        ("최적화 버전", lambda x: quick_sort_optimized(x.copy()))
    ]
    
    for name, func in algorithms:
        start_time = time.time()
        result = func(large_data.copy())
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.4f}초")