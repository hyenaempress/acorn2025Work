# 버블 정렬 (Bubble Sort)
# 인접한 두 원소를 비교하여 큰 값을 뒤로 보내는 정렬 알고리즘

# 원본 코드의 오류 수정
def bubble_sort_fixed(a):
    """
    원본 코드 수정 버전
    - 오타 수정: a[a +1] → a[i+1]
    - changed 플래그 설정 누락 수정
    """
    n = len(a)
    while True:
        changed = False
        for i in range(0, n - 1):
            if a[i] > a[i+1]:
                print(f"교환 전: {a}")
                a[i], a[i+1] = a[i+1], a[i]  # 오타 수정
                print(f"교환 후: {a}")
                changed = True  # 교환이 일어났음을 표시
        if changed == False:
            return

# 방법 1: 원리 이해를 우선시 (단계별 출력)
def bubble_sort_educational(arr):
    """
    교육용 버블 정렬 - 각 단계를 자세히 보여줌
    """
    arr = arr.copy()  # 원본 보호
    n = len(arr)
    print(f"정렬 시작: {arr}")
    
    for pass_num in range(n - 1):  # n-1번의 패스
        print(f"\n--- {pass_num + 1}번째 패스 ---")
        swapped = False
        
        for i in range(n - 1 - pass_num):  # 매 패스마다 범위 축소
            print(f"비교: {arr[i]} vs {arr[i+1]}", end=" ")
            
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                swapped = True
                print(f"→ 교환! {arr}")
            else:
                print("→ 그대로")
        
        if not swapped:  # 교환이 없으면 정렬 완료
            print("교환이 없었으므로 정렬 완료!")
            break
    
    print(f"\n최종 결과: {arr}")
    return arr

# 방법 2: 실무에서 사용하는 최적화된 버전
def bubble_sort_optimized(arr):
    """
    최적화된 버블 정렬
    - 각 패스마다 정렬 범위 축소
    - 조기 종료 (early termination)
    """
    n = len(arr)
    
    for i in range(n - 1):
        swapped = False
        last_swapped_index = 0
        
        # 매번 정렬된 부분은 제외
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                last_swapped_index = j
        
        # 교환이 없으면 정렬 완료
        if not swapped:
            break
    
    return arr

# 방법 3: 칵테일 정렬 (양방향 버블 정렬)
def cocktail_sort(arr):
    """
    칵테일 정렬 (Cocktail Sort) - 버블 정렬의 개선 버전
    - 양방향으로 버블링하여 성능 개선
    """
    n = len(arr)
    left = 0
    right = n - 1
    
    while left < right:
        # 왼쪽에서 오른쪽으로 (큰 값을 오른쪽으로)
        for i in range(left, right):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
        right -= 1
        
        # 오른쪽에서 왼쪽으로 (작은 값을 왼쪽으로)
        for i in range(right, left, -1):
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
        left += 1
    
    return arr

# 성능 비교를 위한 다른 정렬 알고리즘들
def selection_sort(arr):
    """선택 정렬 - 비교용"""
    n = len(arr)
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    """삽입 정렬 - 비교용"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 테스트 및 성능 비교
if __name__ == "__main__":
    # 원본 코드 테스트 (수정 버전)
    print("=== 원본 코드 수정 버전 ===")
    d = [2, 4, 5, 1, 3]
    bubble_sort_fixed(d.copy())
    print(f"결과: {d}")
    
    print("\n=== 교육용 버전 (단계별 출력) ===")
    bubble_sort_educational([2, 4, 5, 1, 3])
    
    print("\n=== 다양한 정렬 알고리즘 비교 ===")
    test_arrays = [
        [64, 34, 25, 12, 22, 11, 90],
        [5, 1, 4, 2, 8],
        [1, 2, 3, 4, 5],  # 이미 정렬된 경우
        [5, 4, 3, 2, 1],  # 역순 정렬된 경우
    ]
    
    algorithms = [
        ("버블 정렬", bubble_sort_optimized),
        ("칵테일 정렬", cocktail_sort),
        ("선택 정렬", selection_sort),
        ("삽입 정렬", insertion_sort),
    ]
    
    for i, test_data in enumerate(test_arrays):
        print(f"\n테스트 {i+1}: {test_data}")
        for name, func in algorithms:
            result = func(test_data.copy())
            print(f"{name:10}: {result}")
    
    print("\n=== 시간 복잡도 분석 ===")
    complexities = {
        "버블 정렬": {
            "최선": "O(n)",      # 이미 정렬된 경우
            "평균": "O(n²)", 
            "최악": "O(n²)",     # 역순 정렬된 경우
            "공간": "O(1)"
        },
        "선택 정렬": {
            "최선": "O(n²)",
            "평균": "O(n²)",
            "최악": "O(n²)",
            "공간": "O(1)"
        },
        "삽입 정렬": {
            "최선": "O(n)",
            "평균": "O(n²)",
            "최악": "O(n²)",
            "공간": "O(1)"
        }
    }
    
    for algo, complexity in complexities.items():
        print(f"\n{algo}:")
        for case, value in complexity.items():
            print(f"  {case:4}: {value}")
    
    # 실제 성능 측정
    print("\n=== 실제 성능 측정 (1000개 랜덤 데이터) ===")
    import random
    import time
    
    large_data = [random.randint(1, 1000) for _ in range(1000)]
    
    # 버블정렬은 느리므로 작은 데이터로 테스트
    small_data = [random.randint(1, 100) for _ in range(100)]
    
    performance_algorithms = [
        ("버블 정렬", bubble_sort_optimized),
        ("선택 정렬", selection_sort),
        ("삽입 정렬", insertion_sort),
    ]
    
    for name, func in performance_algorithms:
        start_time = time.time()
        func(small_data.copy())
        end_time = time.time()
        print(f"{name:10}: {end_time - start_time:.6f}초")
    
    print("\n버블정렬의 특징:")
    print("✓ 구현이 간단함")
    print("✓ 안정 정렬 (같은 값의 순서 유지)")
    print("✓ 제자리 정렬 (추가 메모리 불필요)")
    print("✗ 성능이 느림 O(n²)")
    print("✗ 실무에서는 거의 사용하지 않음")
    print("→ 주로 교육용이나 매우 작은 데이터에만 사용")