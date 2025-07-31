#기본 통계 함수를 직접 작성, 평균 분산, 표준편차 구하기 

grades = [1, 3, -2, 4]
def grades_sum(grades):
    total = 0
    for g in grades:
        total += g
    return total

#평균 구하기 
print(grades_sum(grades))

def grades_average(grades):
    return grades_sum(grades) / len(grades)
print(grades_average(grades))

#분산 구하기
def grades_variance(grades):
    avg = grades_average(grades)
    total = 0
    for g in grades:
        total += (g - avg) ** 2
    return total / len(grades)
print(grades_variance(grades))

#표준편차 구하기
def grades_stddev(grades):
    return grades_variance(grades) ** 0.5       
print(grades_stddev(grades))

#R 가지고 하는거랑 표준 편차 다르게 나옴 자유도의 차이 때문임 
#분산과 표준 편자 값이 최대한 일치하도록 수정 해야함 
#자유도 적용한 분산 구하기
#자유도는 표본의 수에서 1을 뺀 값으로 계산
#자유도 적용한 분산 구하기
#자유도는 표본의 수에서 1을 뺀 값으로 계산

def grades_variance(grades):
    avg = grades_average(grades)
    total = 0
    for g in grades:
        total += (g - avg) ** 2
    return total / (len(grades) - 1)  # 자유도 적용
print(grades_variance(grades))                  

#numpy로 표준편차 구하기
import numpy as np
grades = np.array([1, 3, -2, 4])     
print(np.std(grades, ddof=1))  # 자유도 1로 표준편차 계산
#numpy로 분산 구하기
print(np.var(grades, ddof=1))  # 자유도 1로 분산 계산
#numpy로 평균 구하기
print(np.mean(grades))  # 평균 계산    
#가중평균이란?
#각 데이터에 가중치를 곱한 후 평균을 구하는 방법
#가중치가 있는 데이터의 평균을 구할 때 사용   
#언제 예시로는 학생의 성적을 계산할 때, 각 과목의 중요도에 따라 가중치를 부여하여 평균을 구하는 경우가 있습니다.
#numpy로 가중평균 구하기
weights = np.array([0.1, 0.2, 0.3, 0.4])  # 가중치 배열
print(np.average(grades, weights=weights))  # 가중평균 계산        
#numpy로 합계 구하기
print(np.sum(grades))  # 합계 계산  
#numpy로 최대값 구하기
print(np.max(grades))  # 최대값 계산            
#numpy로 최소값 구하기      
print(np.min(grades))  # 최소값 계산    
#numpy로 중앙값 구하기
print(np.median(grades))  # 중앙값 계산 
#numpy로 사분위수 구하기
print(np.percentile(grades, 25))  # 1사분위수   
print(np.percentile(grades, 50))  # 2사분위수 (중앙값)
print(np.percentile(grades, 75))  # 3사분위수                   

