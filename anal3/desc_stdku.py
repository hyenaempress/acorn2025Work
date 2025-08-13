# 표준편차, 분산은 중요하다.
# 2개 반의 시험 성적이 다를 때, 그 차이를 수치적으로 나타내기 위해 사용된다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

np.random.seed(42) # 시드 넘버 고정: 재현 가능성 확보

# 목표 평균...
target_mean = 60 # 목표 평균
std_dev_small = 10 # 표준편차 최소 값
std_dev_large = 20 # 표준편차 최대 값

class1_raw = np.random.normal(loc = target_mean, scale = std_dev_small, size = 100) # 1반, 평균에 몰려있을 것으로 예상
class2_raw = np.random.normal(loc = target_mean, scale = std_dev_large, size = 100) # 2반, 1반에 비하여 흩어져있을 거라 예상

# 평균 보정
class1_adj = class1_raw - np.mean(class1_raw) + target_mean
class2_adj = class2_raw - np.mean(class2_raw) + target_mean

# 정수화 및 범위 제한
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)

print("데이터 1차 가공 결과")
print("class1: \n", class1)
print("class2: \n", class2)

# 통계 계산
mean1, mean2 = np.mean(class1), np.mean(class2) # 평균
std1, std2 = np.std(class1), np.std(class2) # 표준편차
var1, var2 = np.var(class1), np.var(class2) # 분산

# 출력
print("1반 성적: ", class1)
print("평균 = {:.2f}, 표준편차 = {:.2f}, 분산 = {:.2f}".format(mean1, std1, var1))
print("2반 성적: ", class2)
print("평균 = {:.2f}, 표준편차 = {:.2f}, 분산 = {:.2f}".format(mean2, std2, var2))

# 데이터프레임
df = pd.DataFrame({
    'Class':['1반'] * 100 + ['2반'] * 100,
    'Score': np.concatenate([class1, class2])
})

print(df)

22.04 카이제곱 검정과 유의확률
22.05 t검정 완전 가이드
22.06 ANOVA (분산분석)
22.07 회귀분석 기초

📊 5단계: 고급 통계
22.08 비모수 검정
22.09 베이지안 통계 입문
22.10 실무 통계 프로젝트