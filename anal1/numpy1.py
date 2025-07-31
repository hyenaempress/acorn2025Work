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
#분산과 표준 편자 값이 최대한 일치하도록 수정
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
