#삽입 정렬은 자료의 배열의 모든 요소를 앞에서부터 차례대로 이미 정렬된 
#배열 부분과 비교하여, 자신의 위치를 찾아 삽임함으로서 정렬을 완성하는 알고리즘이다.
#알고리즘 과정 

#방법 1 원리를 이해를 우선 
def find_insFunc(r, v):
    #이미 정렬된 r자료를 앞에서 부터 차례로 확인
    for i in range(0, len(r)):
        if v < r[i]:
            return i
    return len(r) #v가 r의 모든 요소값 보다 클 경우레은 맨 뒤에 삽입 


def ins_sort(a):
    result = []
    while a:
        value = a.pop(0)
        ins_idx = find_insFunc(result, value)
        result.insert(ins_idx, value)  #찾은 위치에 값 삽입 또는 추가 (이후 값은 밀려남)
    return result

d = [2, 4, 5, 1, 3]
print(ins_sort(d))  
        

#방법 2번 

def ins_sort2(a):
    n = len(a)
    for i in range(1, n ): #두번째 값 (인덱스 1) 부터 마지막까지 차례대로 '삽입할 대상' 선택 
        key = a[i] 
        j = i -1 
        while j >= 0 and a[j] > key : #key 값보다 큰 값을 우측으로 밀기 (참)
            a[j + 1] = a[j]
            j -= 1
        a[j+1] = key
    
d = [2, 4, 5, 1, 3]
print(ins_sort2(d))  

#버블 vs 선택 vs 퀵 정렬을 놓고,
#시간복잡도 변화
#비교 위치/교환 방식
#반복문 구조
#분할여부 (Divide & Conquer 유무)
#알고리즘 개념으로 밈, 스토리텔링, 캐릭터화 (이거 ADHD에 진짜 잘 먹힘)