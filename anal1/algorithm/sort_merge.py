# 합병 정렬(Merge Sort)은 주어진 데이터 리스트를 분할하고 정복하여 정렬하는 알고리즘
# 알고리즘 상세 설명:
# 1. 주어진 리스트를 반으로 나눈다
# 2. 각 부분 리스트를 재귀적으로 합병 정렬한다
# 3. 정렬된 두 부분 리스트를 하나로 합병하여 최종 정렬된 리스트를 만든다

# 합병 정렬 방법 1: 새로운 리스트(result)를 만들어 정렬 (메모리 사용 많음)
# 시간 복잡도: O(n log n)
# 공간 복잡도: O(n)

def merge_sort(a):
    n = len(a)
    if n <= 1:
        return a
    
    mid = n // 2  # = 기호 추가
    g1 = merge_sort(a[:mid])  # 띄어쓰기 수정
    g2 = merge_sort(a[mid:])  # 띄어쓰기 수정
    
    # 핵심! 병합 과정 추가
    return merge(g1, g2)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 테스트
d = [6, 8, 3, 1, 2, 4, 7, 5]
print(merge_sort(d))  # [1, 2, 3, 4, 5, 6, 7, 8]
# 합병 정렬 방법 2: 제자리 정렬 (실제 문제 풀이에 적합)
# 시간 복잡도: O(n log n)
# 공간 복잡도: O(1)
def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

#합병 정렬 방법 3 : 

