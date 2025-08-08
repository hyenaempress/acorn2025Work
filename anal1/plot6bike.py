#자전거 공유 시스템(워싱턴 D.C) 관련 파일로 시각화 
#리니어 리그레이션

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False
#다른 방법도 있는데 이런식으로 쓰는 것이다 만약 주피터로 썻으면 맥플로립 명령어 

plt.style.use('ggplot')
#데이터 수집 후 가곡 (EDA) 을 해야하지만 어느정도 데이터 가공이 되어있는것이다. 

train=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', parse_dates=['datetime'])
print(train.shape)
print(train.columns)
print(train.info()) # 각 구조를 보는것이 중요합니다. 
print(train.head(3))

print(train.describe()) #전체적인 요약 통계량을 볼 수 있씁니다. 
pd.set_option('display.max_columns', 500)
#print(train.describe())
print(train.temp.head(3))
print(train.isnull().sum()) #결측치 확인  널이 포함된 열 확인용 시각화 모듈이 따로있어요! 근데 안깔려 있어요 

#null 이 포함된 포함을 검색하는 missingno 모듈을 써보겠습니다.

#import missingno as msno
#msno.matrix(train, figsize=(12,5)) #결측치 확인용 시각화 모듈이 따로있어요! 
#plt.show() 
#msno.bar(train) 
#plt.show() 

#이런식으로 볼 수 있어요 

#연월일시 데이터로 자전거 대여량 시각화 
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second

print(train.columns)
print(train.head(1))

figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)#이게 객체 지향적인 방법이다. 
figure.set_size_inches(15, 5)
sns.barplot(x='year', y='count', data=train, ax=ax1)
sns.barplot(x='month', y='count', data=train, ax=ax2)
sns.barplot(x='day', y='count', data=train, ax=ax3)
sns.barplot(x='hour', y='count', data=train, ax=ax4)

ax1.set(ylabel='건수', title='연도별 자전거 대여량')
ax2.set(ylabel='건수', title='월별 자전거 대여량')
ax3.set(ylabel='건수', title='일별 자전거 대여량')
ax4.set(ylabel='건수', title='시간별 자전거 대여량')

plt.show()


#챕터 2 Boixplot으로 시각화 - 대여량 - 계졀별, 시간별 근무일 여부에 따른 대여량 

figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(15, 5)
sns.boxplot(x='hour', y='count', data=train, ax=ax1)
sns.boxplot(x='hour', y='count', data=train, ax=ax2)
ax1.set(ylabel='건수', title='시간별 자전거 대여량')
ax2.set(ylabel='건수', title='시간별 자전거 대여량')
plt.show()



