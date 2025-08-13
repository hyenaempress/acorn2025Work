# 기술 통계의 목적은 데이터를 수집, 요약, 정리, 시각화
# 도수분포표(Frequency Distribution Table)는 데이터를 구간으로 나누고 각 구간에 속하는 데이터의 개수를 세어 나타낸 표이다.
# 이를 통해 데이터의 분포를 한 눈에 파악할 수 있다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# step 1: 데이터 읽어서 DataFrame에 저장한다.
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv')
print(df.head(2))

# step 2: max, min
min_height = df['키'].min()
max_height = df['키'].max()
print(f'최소 값: {min_height}')
print(f'최대 값: {max_height}') # 최소 값: 158, 최대 값: 191

# step 3: 구간 설정
bins = list(np.arange(156, 195, 5)) # 구간 설정
print("설정된 구간 정보:\n",bins)
df['계급'] = pd.cut(df['키'], bins=bins, right=True)
print(df.head(3))
print(df.tail(3))

# # step 4: 각 계급의 중앙값 (156 + 161) / 2
# df['계급값'] = df['계급'].apply(lambda x: (x.left + x.right) / 2)
# print("step4:\n", df.head(3))

# step4: 각 계급의 중앙값 (156+161)/2 
df['계급값']=df['계급'].apply(lambda x:int(x.left+x.right)/2)
print(df.head(3)) 
print(df.tail(3)) 

# step 5: 도수 계산
freq = df['계급'].value_counts().sort_index()

# step 6: 상대 도수(전체 데이터에 대한 비율) 계산
relative_freq = (freq / freq.sum()).round(2)
print(relative_freq)

# step 7: 누적 도수 계산 - 계급별 도수 누적
cum_freq = freq.cumsum()
print(cum_freq)

# step by step uh uh baby~

# step 8: 도수 분포표 작성
dist_table = pd.DataFrame({
    # "156 ~ 161" 이런 모양 출력하기
    '계급':[f"{int(interval.left)} ~ {int(interval.right)}" for interval in freq.index],
    # 계급의 중간값
    '계급값': [(int(interval.left) + int(interval.right)) / 2 for interval in freq.index],
    # 도수
    '도수': freq.values,
    # 상대 도수
    '상대 도수': relative_freq.values,
    # 누적 도수
    '누적 도수': cum_freq.values
})
print('도수 분포표')
print(dist_table.head(3))
print(dist_table['계급값'])


plt.figure(figsize=(16, 10))
# step 9: 다양한 그래프 그리기
# 1. 기본 막대그래프 (도수)
plt.subplot(2, 2, 1)
print(dist_table)
plt.bar(dist_table['계급값'], dist_table['도수'], width=5, color='cornflowerblue', edgecolor='black')
plt.title('학생 50명 키 히스토그램', fontsize=16)
plt.xlabel('키(계급값)')
plt.ylabel('도수')
plt.xticks(dist_table['계급값'])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 2. 꺾은선그래프 (누적 도수)
plt.subplot(2, 2, 2)
plt.plot(dist_table['계급'], dist_table['누적 도수'], marker='o', color='orange')
plt.title('누적 도수 꺾은선그래프', fontsize=16)
plt.xlabel('계급')
plt.ylabel('누적 도수')
plt.xticks(rotation=45)

# 3. 상대 도수 막대그래프
plt.subplot(2, 2, 3)
plt.bar(dist_table['계급'], dist_table['상대 도수'], color='lightgreen', edgecolor='black')
plt.title('상대 도수 막대그래프', fontsize=16)
plt.xlabel('계급')
plt.ylabel('상대 도수')
plt.xticks(rotation=45)

# 4. 도수 분포 파이차트
plt.subplot(2, 2, 4)
plt.pie(dist_table['도수'], labels=list(dist_table['계급']), autopct='%1.1f%%', startangle=90)
plt.title('도수 분포 파이차트', fontsize=16)

plt.tight_layout()
plt.show()