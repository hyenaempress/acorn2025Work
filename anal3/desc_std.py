#표준편차, 분산을 중요
#두 반의 시험 성적이 평균이 같다고 해서성적 분포가 동일한가?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')

np.random.seed(42)
#목표 평균 
target_mean = 60
std_dev_small = 10
std_dev_large = 20

class1_raw = np.random.normal(target_mean, std_dev_small, 100)
class2_raw = np.random.normal(target_mean, std_dev_small, 100)