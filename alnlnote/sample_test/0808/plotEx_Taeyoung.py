import pandas as pd
import numpy as np
from pandas import DataFrame
from bs4 import BeautifulSoup
import urllib
import requests
import matplotlib.pyplot as plt

plt.rc('font', family = 'malgun gothic') # 한글 깨짐 방지 
plt.rcParams['axes.unicode_minus'] = False # 한글 깨짐 방지 

url = 'https://www.goobne.co.kr/menu/menu_list_p'
response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'lxml')
# print(soup)

names = [tag.text.strip() 
         for tag in soup.select('h4')
         ]
print(names)

prices = [int(tag.text.strip().replace(',','')) 
          for tag in soup.select('b')
          ]
print(prices)

df = pd.DataFrame({'메뉴': names, '가격': prices})
print(df)
print('가격평균: ', round(df['가격'].mean(),2), '원')
print(f"가격 평균: {df['가격'].mean():.2f}")
print('가격 표준편차: ', round(df['가격'].std(),2))

def classify(row):
    if row['가격'] > 16000:
        if '피자' in row['메뉴']:
            return '피자'
        else:
            return '치킨'
    elif row['가격'] <15000:
        return '사이드메뉴'
df['분류'] = df.apply(classify, axis = 1)

chicken = df[df['분류'] == '치킨']
pizza = df[df['분류'] == '피자']
side = df[df['분류'] == '사이드메뉴']

print('치킨 메뉴:\n',chicken)
print('피자 메뉴:\n',pizza)
print('사이드 메뉴:\n',side)

plt.xlabel('메뉴이름')
plt.ylabel('가격 (단위 : 원)')
plt.figure(figsize = (8,10))
plt.title('굽네치킨 전 메뉴')

plt.subplot(2,2,1)
plt.xlabel('메뉴이름')
plt.ylabel('가격 (단위 : 원)')
plt.plot(chicken['메뉴'],chicken['가격'],'o-')
plt.xticks(rotation = 45, ha = 'right')

plt.subplot(2,2,2)
plt.xlabel('메뉴이름')
plt.ylabel('가격 (단위 : 원)')
plt.plot(pizza['메뉴'], pizza['가격'],'bo-')
plt.xticks(rotation = 45, ha = 'right')

plt.subplot(2,2,(3,4))
plt.xlabel('메뉴이름')
plt.ylabel('가격 (단위 : 원)')
plt.plot(side['메뉴'],side['가격'],'go-')
plt.xticks(rotation = 45, ha = 'right')

plt.tight_layout
plt.show()

# 출력
# 메뉴명	가격	설명

# 건수:
# 가격평균:
# 표준편차:
# 최고가격:
# 최저가격:
# 기타 ...
# 시각화