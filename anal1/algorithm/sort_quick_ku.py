# 퀵 정렬(Quick Sort)은 주어진 데이터 리스트를 분할하고 정복하여 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 피벗(pivot)을 선택한다 (일반적으로 첫 번째 원소나 마지막 원소)
# 2. 피벗을 기준으로 리스트를 두 부분으로 나눈다 (피벗보다 작은 값과 큰 값)
# 3. 각 부분 리스트를 재귀적으로 퀵 정렬한다
# 4. 정렬된 두 부분 리스트를 합쳐 최종 정렬된 리스트를 만든다

# 퀵 정렬 방법 1: 비파괴적(새 리스트 반환, 슬라이싱 기반)
# 시간 복잡도: O(n log n) (평균), O(n^2) (최악)
# 공간 복잡도: O(n) (새 리스트 생성)
def quick_sort1(list_arguement, verbose=False):
    length = len(list_arguement)

    if length <= 1:  # 리스트가 1개 이하이면 이미 정렬된 상태
        return list_arguement
    
    # 기준 값(피벗) 선택: 마지막 원소
    pivot = list_arguement[length - 1]

    # 피벗을 기준으로 리스트를 두 부분으로 나누기
    less_than_pivot = []      # 피벗보다 작은 값들
    greater_than_pivot = []   # 피벗보다 크거나 같은 값들
    for i in range(length - 1):  # 마지막 원소는 피벗이므로 제외
        if list_arguement[i] < pivot:
            less_than_pivot.append(list_arguement[i])
        else:
            greater_than_pivot.append(list_arguement[i])
    if verbose:
        print(f"피벗: {pivot}, 리스트: {list_arguement}")

    # 재귀적으로 정렬 후 결과 합치기
    return quick_sort1(less_than_pivot, verbose) + [pivot] + quick_sort1(greater_than_pivot, verbose)

# 퀵 정렬 방법 2: 제자리(in-place) 정렬 (원본 리스트 직접 변경)
# 시간 복잡도: O(n log n) (평균), O(n^2) (최악)
# 공간 복잡도: O(1) (추가 메모리 거의 없음)
def quick_sort_sub(list_arguement, start, end, verbose=False):
    # 종료 조건: 정렬 대상이 한 개 이하이면 정렬하지 않음
    if end - start <= 0:
        return
    
    # 피벗 선택: 마지막 원소
    pivot = list_arguement[end]
    i = start
    # 피벗보다 작은 값은 왼쪽, 큰 값은 오른쪽으로 이동
    for j in range(start, end):
        if list_arguement[j] < pivot:
            # i 자리에 작은 값을 넣고, j 자리에 큰 값을 넣음
            list_arguement[i], list_arguement[j] = list_arguement[j], list_arguement[i]
            i += 1
    # 피벗을 i 자리에 넣음 (피벗 기준으로 분할 완료)
    list_arguement[i], list_arguement[end] = list_arguement[end], list_arguement[i]

    # 재귀 호출: 피벗 기준 왼쪽/오른쪽 부분 정렬
    quick_sort_sub(list_arguement, start, i - 1, verbose)  # 피벗 왼쪽 부분 정렬
    quick_sort_sub(list_arguement, i + 1, end, verbose)    # 피벗 오른쪽 부분 정렬

    if verbose:
        print(f"정렬 중: {list_arguement}")

def quick_sort2(list_arguement, verbose=False):
    """
    퀵 정렬 방법 2: 제자리(in-place) 정렬
    원본 리스트를 직접 정렬하며, 반환값은 정렬된 리스트(자기 자신)
    """
    quick_sort_sub(list_arguement, 0, len(list_arguement) - 1, verbose)
    return list_arguement  # 추가 메모리 사용 없이 원본 리스트를 직접 정렬함