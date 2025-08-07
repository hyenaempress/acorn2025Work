# 웹 스크래핑
from bs4 import BeautifulSoup
import urllib.request # 연습용, 코드가 장황 
import requests # 실전용, 코드가 간결
import pandas as pd

url = "https://www.kyochon.com/menu/chicken.asp"
response = requests.get(url)
response.raise_for_status()  # 요청이 성공했는지 확인

soup = BeautifulSoup(response.text, 'html.parser')
print(soup)

# 메뉴 이름 추출
names = [tag.text.strip() for tag in soup.select('dl.txt>dt')]
print(names)

# 가격 이름 추출
prices = [int(tag.text.strip().replace(',', '')) for tag in soup.select('p.money strong')]
print(prices)

# 메뉴 이름과 가격을 딕셔너리로 묶기
df = pd.DataFrame({
    '상품명': names,
    '가격': prices
})
print(df)
print('가격 평균', round(df['가격'].mean(), 2))

print((f"가격 평균: {df['가격'].mean():.2f}"))

print('가격 표준편차', round(df['가격'].std(), 2))

print((f"가격 표준편차: {df['가격'].std():.2f}"))

#XML  은 부모와 자식관계를 알하야한다는 부담이 있습니다 그래서 XML은 자주 쓰지 않아요. 