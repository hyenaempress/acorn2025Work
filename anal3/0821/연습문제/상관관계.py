# 상관관계 문제

# https://github.com/pykwon/python 에 있는 Advertising.csv 파일을 읽어 tv,radio,newspaper 간의 상관관계를 파악하시오. 
# 그리고 이들의 관계를 heatmap 그래프로 표현하시오. 

import pandas as pd                # pandas 라이브러리 불러오기
import numpy as np                 # numpy 라이브러리 불러오기
import seaborn as sns              # seaborn 라이브러리 불러오기
import matplotlib.pyplot as plt    # matplotlib 라이브러리 불러오기

df = pd.read_csv("./01_big_data_machine_learning/data/Advertising.csv") # Advertising.csv 파일 읽기
print(df)                         # 데이터프레임 출력

df = df[['tv', 'radio', 'newspaper']]         # tv, radio, newspaper 컬럼만 선택
corr = df.corr()                             # 선택한 컬럼들 간의 상관계수 행렬 계산
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f') # 상관계수 행렬을 히트맵으로 시각화
plt.show()                                   # 그래프