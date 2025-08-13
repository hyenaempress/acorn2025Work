#기술 통계의 목적을 데이터를 수집 요약 정리 시각화 
#도수분포표 는 데이터를 구간별로 나눠 빈도를 계산하는 표 
#이를 통해 데이터의 분포를 한 눈에 파악 할 수 있다

import pandas as pd
#충분히 알만한 내용이지만 하고 지나갑시다. 
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

#step 1 : 데이터를 읽어서 데이터 프레임에 저장하는 것 

df= pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv', encoding='utf-8') 
print(df.head(2))

#step 2 : 최대값 구하고 최소값 구하고

min_height = df['키'].min()
max_height = df['키'].max()
print(min_height, max_height)

#step 3 : 키의 범위를 구하고 구간을 나누고 

height_range = max_height - min_height
print(height_range)

#step 4 : 구간을 나누고 
bins = np.arange(min_height, max_height+4, 5)
print(bins)

df['계급'] = pd.cut(df['키'], bins, right=False, include_lowest=True)
print(df.head(2))

#step 5 : 도수분포표 구하고 
freq_table = pd.crosstab(index=df['계급'], columns='count')
print(freq_table)

#step 6 : 도수분포표를 시각화 하고 
freq_table.plot(kind='barh', color='skyblue', width=0.8)
plt.show()

#step 7 : 중앙값을 구하고
median_height = np.median(df['키'])
print(median_height)

#step 8 : 평균값을 구하고
mean_height = np.mean(df['키'])
print(mean_height)

#step 9 : 최빈값을 구하고
mode_height = df['키'].mode()
print(mode_height)

#step 10 : 표준편차를 구하고
std_height = np.std(df['키'])
print(std_height)

#step 11 : 분산을 구하고
var_height = np.var(df['키'])
print(var_height)

#step 12 : 데이터 프레임에 추가하고
df['중앙값'] = median_height
df['평균값'] = mean_height
df['최빈값'] = mode_height
df['표준편차'] = std_height

#step 13 : 데이터 프레임을 출력하고
print(df.head(2))

#step 14 : 데이터 프레임을 시각화 하고
df.plot(kind='barh', color='skyblue', width=0.8)
plt.show()


#좀더 잘 보이는 표로 바꾸고  
