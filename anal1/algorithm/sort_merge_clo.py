# 합병 정렬(Merge Sort) 구현
# 분할 정복 알고리즘을 사용하여 O(n log n) 시간복잡도로 정렬

# 방법 1: 새로운 리스트를 반환하는 방식 (함수형 접근)
# 시간 복잡도: O(n log n), 공간 복잡도: O(n)

def merge_sort_functional(arr):
    """
    새로운 정렬된 리스트를 반환하는 합병 정렬
    원본 리스트는 변경되지 않음
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort_functional(arr[:mid])
    right = merge_sort_functional(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """두 정렬된 리스트를 병합하여 하나의 정렬된 리스트 반환"""
    result = []
    i = j = 0
    
    # 두 리스트를 비교하며 작은 값부터 result에 추가
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 남은 요소들 추가
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


# 방법 2: 제자리 정렬 방식 (In-place sorting)
# 시간 복잡도: O(n log n), 공간 복잡도: O(log n) - 재귀 호출 스택

def merge_sort_inplace(arr, left=0, right=None):
    """
    원본 리스트를 직접 수정하는 합병 정렬
    메모리 효율적이지만 추가 배열이 필요함
    """
    if right is None:
        right = len(arr) - 1
    
    if left < right:
        mid = (left + right) // 2
        
        # 왼쪽과 오른쪽 부분을 재귀적으로 정렬
        merge_sort_inplace(arr, left, mid)
        merge_sort_inplace(arr, mid + 1, right)
        
        # 정렬된 두 부분을 병합
        merge_inplace(arr, left, mid, right)

def merge_inplace(arr, left, mid, right):
    """제자리에서 두 정렬된 부분을 병합"""
    # 임시 배열 생성
    temp = arr[left:right + 1]
    
    i = 0  # 왼쪽 부분의 시작 인덱스
    j = mid - left + 1  # 오른쪽 부분의 시작 인덱스
    k = left  # 원본 배열의 현재 위치
    
    # 병합 과정
    while i <= mid - left and j <= right - left:
        if temp[i] <= temp[j]:
            arr[k] = temp[i]
            i += 1
        else:
            arr[k] = temp[j]
            j += 1
        k += 1
    
    # 남은 요소들 복사
    while i <= mid - left:
        arr[k] = temp[i]
        i += 1
        k += 1
    
    while j <= right - left:
        arr[k] = temp[j]
        j += 1
        k += 1


# 방법 3: 디버깅 정보를 포함한 교육용 버전
def merge_sort_verbose(arr, depth=0, verbose=False):
    """
    정렬 과정을 시각적으로 보여주는 합병 정렬
    교육 목적으로 사용하기 좋음
    """
    indent = "  " * depth
    
    if verbose:
        print(f"{indent}정렬 시작: {arr}")
    
    if len(arr) <= 1:
        if verbose:
            print(f"{indent}기저 사례: {arr}")
        return arr
    
    mid = len(arr) // 2
    
    if verbose:
        print(f"{indent}분할: {arr[:mid]} | {arr[mid:]}")
    
    left = merge_sort_verbose(arr[:mid], depth + 1, verbose)
    right = merge_sort_verbose(arr[mid:], depth + 1, verbose)
    
    result = merge(left, right)
    
    if verbose:
        print(f"{indent}병합 결과: {result}")
    
    return result


# 테스트 및 사용 예제
if __name__ == "__main__":
    # 테스트 데이터
    test_data = [6, 8, 3, 1, 2, 4, 7, 5]
    
    print("=== 합병 정렬 테스트 ===")
    print(f"원본 데이터: {test_data}")
    print()
    
    # 방법 1: 함수형 접근
    print("1. 함수형 합병 정렬:")
    sorted_data1 = merge_sort_functional(test_data.copy())
    print(f"결과: {sorted_data1}")
    print()
    
    # 방법 2: 제자리 정렬
    print("2. 제자리 합병 정렬:")
    test_data2 = test_data.copy()
    merge_sort_inplace(test_data2)
    print(f"결과: {test_data2}")
    print()
    
    # 방법 3: 상세 과정 출력
    print("3. 상세 과정을 보여주는 합병 정렬:")
    sorted_data3 = merge_sort_verbose(test_data.copy(), verbose=True)
    print(f"최종 결과: {sorted_data3}")
    
    # 성능 비교를 위한 큰 데이터 테스트
    import random
    import time
    
    print("\n=== 성능 테스트 (10,000개 요소) ===")
    big_data = [random.randint(1, 1000) for _ in range(10000)]
    
    # 함수형 방식 테스트
    start = time.time()
    merge_sort_functional(big_data.copy())
    func_time = time.time() - start
    
    # 제자리 정렬 방식 테스트
    start = time.time()
    test_big = big_data.copy()
    merge_sort_inplace(test_big)
    inplace_time = time.time() - start
    
    print(f"함수형 방식: {func_time:.4f}초")
    print(f"제자리 정렬: {inplace_time:.4f}초")