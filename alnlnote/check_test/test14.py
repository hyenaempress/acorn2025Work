import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (한글 깨짐 방지)
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

#정규분포를 따르는 데이터 1000개 생성 (조건 평균 0 표준편차 1 )
data = np.random.normal(0, 1, 1000)

# 히스토그램 그리기
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('값')
plt.ylabel('빈도')
plt.grid(True, alpha=0.3)
plt.show()

