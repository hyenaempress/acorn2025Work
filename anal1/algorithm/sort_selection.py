#선택 정렬 
#주어진 데이터 리스트에서 가장 작은 원소를 선택하여 맨 앞으로 가져오기
#알고리즘 과정 
#최소값 찾기 : 정렬되지 않은 부분에서 가장 작은 값을 찾습니다.
#교환 : 찾은 최소값을 정렬되지 않은 부분의 맨 앞으로 이동시킵니다.
#반복 : 정렬되지 않은 부분 크기가 1이 될 때 까지 위 과정을 반보합니다.


#방법 1: 원리 이해를 우선시 
def find_minFunc(a):
    n = len(a)
    min_idx = 0
    for i in range(1, n):
        if a[i] < a[min_idx]:
            min_idx = i
    return min_idx  # 들여쓰기 수정

def sel_sort(a):
    a = a.copy()  # 원본 리스트 보호
    result = [] # 지금은 result 를 사용했음 
    while a:
        min_idx = find_minFunc(a)
        value = a.pop(min_idx)
        result.append(value)
    return result

d = [2, 4, 5, 1, 3]
print(find_minFunc(d))  # 3 (인덱스)
print(sel_sort(d))      # [1, 2, 3, 4, 5]
print(d)                # [2, 4, 5, 1, 3] (원본 유지)

#벙법 2 : 일반적 정렬 알고리즘을 구사 
#각 반복마다 작은 값을 해당 집합내의 맨 앞자리와 값을 바꿈 
#이건 꼭 쓸 줄 알아야 합니다. 
def sel_sort2(a):
    n = len(a)
    for i in range(0, n - 1): #0 부터 N - 2 까지 반복한다 
        min_idx = i 
        for j in range(i+1 , n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] =a[min_idx], a[i]
    return a
d = [2, 4, 5, 1, 3]
print(sel_sort2(d))  
#빅오 표기법으로는 ? 

