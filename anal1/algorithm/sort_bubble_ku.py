# 버블 정렬(Bubble Sort)은 인접한 두 원소를 비교하여 정렬하는 알고리즘
# 시간 복잡도: O(n^2) (평균/최악)
# 공간 복잡도: O(1) (제자리 정렬, 추가 메모리 사용 없음)
# 알고리즘 상세 설명:
# 1. 인접한 두 원소를 비교하여 정렬한다.
# 2. 가장 큰 원소가 맨 뒤로 가도록 반복한다.
# 3. 정렬이 완료될 때까지 1~2 단계를 반복한다.
# 4. 최종적으로 정렬된 리스트를 반환한다.
def bubble_sort1(list_arguement, verbose=False):
    length = len(list_arguement)
    if length <= 1:  # 리스트가 1개 이하이면 이미 정렬된 상태
        return list_arguement
    while True:
        changed = False # 변경 여부 플래그, 더 이상 바꿀 값이 없으면 True를 줌
        for i in range(0, length - 1):
            if verbose:
                print(f"비교: {list_arguement[i]}와 {list_arguement[i + 1]}")
            if list_arguement[i] > list_arguement[i + 1]:
                if verbose:
                    print(f"교환: {list_arguement[i]}와 {list_arguement[i + 1]}")
                list_arguement[i], list_arguement[i + 1] = list_arguement[i + 1], list_arguement[i]
                changed = True # 변경이 있었음을 표시
        if changed == False:  # 변경이 없으면 정렬 완료
            if verbose:
                print("더 이상 교환할 값이 없습니다. 정렬 완료.")
            break
    if verbose:
        print(f"최종 정렬 결과: {list_arguement}")
    return list_arguement  # 정렬된 리스트 반환
