# 공분산과 상관계수

import pandas as pd                # pandas 라이브러리 불러오기
import numpy as np                 # numpy 라이브러리 불러오기
import matplotlib.pyplot as plt    # matplotlib의 pyplot 모듈 불러오기
from pandas.plotting import scatter_matrix  # pandas의 산점도 행렬 함수 불러오기
import seaborn as sns              # seaborn 라이브러리 불러오기

plt.rc('font', family='Malgun Gothic') # 한글 폰트 설정

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv") # 데이터 읽기
print(data.head(3))                # 데이터 앞 3개 행 출력
print(data.describe())             # 데이터 요약 통계 출력
print()
print(np.std(data.친밀도))         # 친밀도의 표준편차 출력
print(np.std(data.적절성))         # 적절성의 표준편차 출력
print(np.std(data.만족도))         # 만족도의 표준편차 출력

# plt.hist([np.std(data.만족도), np.std(data.적절성), np.std(data.만족도)])
# plt.show()

print("공분산----------")
print(np.cov(data.친밀도, data.적절성)) # 친밀도와 적절성의 공분산 출력
print(np.cov(data.친밀도, data.만족도)) # 친밀도와 만족도의 공분산 출력
print(data.cov())                       # 전체 변수의 공분산 행렬 출력

print("상관계수----------")
print(np.corrcoef(data.친밀도, data.적절성)) # 친밀도와 적절성의 상관계수 행렬 출력
print(np.corrcoef(data.친밀도, data.만족도)) # 친밀도와 만족도의 상관계수 행렬 출력
print(data.corr())                          # 전체 변수의 상관계수 행렬 출력
print(data.corr(method='pearson'))          # 피어슨 상관계수(등간, 비율 척도) 출력
print(data.corr(method='spearman'))         # 스피어만 상관계수(서열 척도) 출력
print(data.corr(method='kendall'))          # 켄달 상관계수(서열 척도) 출력

# 예) 만족도에 대한 다른 특성(변수) 사이의 상관 관계 보기
co_re = data.corr()                        # 전체 변수의 상관계수 행렬 저장
print(co_re['만족도'].sort_values(ascending=False)) # 만족도 기준 내림차순 정렬하여 출력
# 만족도    1.000000
# 적절성    0.766853
# 친밀도    0.467145

# 시각화
data.plot(kind = 'scatter', x = '만족도', y = '적절성') # 만족도와 적절성의 산점도 그리기
plt.show()                                             # 그래프 표시

attr = ['친밀도', '적절성', '만족도']                   # 분석할 변수 리스트

scatter_matrix(data[attr], figsize=(10, 6))            # 산점도 행렬 그리기
plt.show()                                             # 그래프 표시

sns.heatmap(data.corr())                               # 상관계수 행렬 히트맵 그리기
plt.show()                                             # 그래프 표시

