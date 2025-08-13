# 표준편차, 분산을 중요
# 두 반의 시험 성적이 "평균이 같다고 해서 성적분포가 동일한가?"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

np.random.seed(42)  # 랜덤 시드 설정
# 목표평균 ... 
target_mean = 60
std_dev_small = 10
std_dev_large = 20

class1_raw = np.random.normal(loc=target_mean, scale=std_dev_small, size=100)
class2_raw = np.random.normal(loc=target_mean, scale=std_dev_large, size=100)

# 평균 보정
class1_adj = (class1_raw - np.mean(class1_raw)) + target_mean
class2_adj = (class2_raw - np.mean(class2_raw)) + target_mean

# 정수화 및 범위 제한
class1 = np.clip(np.round(class1_adj), 10, 100).astype(int)
class2 = np.clip(np.round(class2_adj), 10, 100).astype(int)

print(class1)
print(class2)

# 통계값 계산
mean1, mean2 = np.mean(class1), np.mean(class2)
std1, std2 = np.std(class1), np.std(class2)
var1, var2 = np.var(class1), np.var(class2)

print(f"Class 1 -\n Mean(평균): {mean1}\n Std Dev(표준편차): {std1:.2f}\n Variance(분산): {var1:.2f}")
print(f"Class 2 -\n Mean(평균): {mean2}\n Std Dev(표준편차): {std2:.2f}\n Variance(분산): {var2:.2f}")

df = pd.DataFrame({
    "Class": ["1반"] * 100 + ["2반"] * 100,
    "score": np.concatenate([class1, class2]),
})

print(df.head(3))
print(df.tail(3))