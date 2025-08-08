from bs4 import BeautifulSoup
import urllib.request as req
import pandas as pd
import numpy as np
import re

url = "https://m.popmart.co.kr/product/list.html?cate_no=392"
response = req.urlopen(url)
html = response.read().decode('utf-8')
soup = BeautifulSoup(html, 'html.parser')

# 상품명
names = [tag.text.strip() for tag in soup.select('p.name')]

# 상품가격
prices = [tag.text.strip() for tag in soup.select('span.price')]

# 문자열 가격 → 숫자 변환
price_nums = []
for price in prices:
    price_num = re.sub(r'[^0-9]', '', price)
    if price_num:
        price_nums.append(int(price_num))

# 출력: 상품명 + 가격
for name, price in zip(names, prices):
    print(f"상품명: {name} | 가격: {price}")

# numpy 통계 처리
price_array = np.array(price_nums)
avg_price = price_array.mean()
std_price = price_array.std()
max_price = price_array.max()

# 최고가 상품
expensive_items = [name for name, price in zip(names, price_nums) if price == max_price]

# 통계 출력
print(f"\n총 상품 수: {len(names)}개")
print(f"평균 가격: {avg_price:,.0f}원")
print(f"가격 표준편차: {std_price:,.0f}원")
print(f"최고 가격: {max_price:,}원")
print("💰 최고가 상품:")
for item in expensive_items:
    print(f" - {item}")
