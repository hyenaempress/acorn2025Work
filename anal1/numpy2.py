#numpy 기본 기능
#기본 통계 함수를 직접 작성, 평균 분산, 표준편차 구하기
import numpy as np
grades = np.array([1, 3, -2, 4])
def grades_sum(grades):
    total = 0
    for g in grades:
        total += g
    return total
