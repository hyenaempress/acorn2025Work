from bs4 import BeautifulSoup
import urllib.request as req
import re
import numpy as np

#페이지 순환 스크래핑 예제 

base_url = "https://m.popmart.co.kr/product/list.html?cate_no=392"
page = 1

all_names = []
all_prices = []

while True: #페이지 순환 루프 추가 
    url = f"{base_url}&page={page}"
    response = req.urlopen(url)
    html = response.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')


    names = [tag.text.strip() for tag in soup.select('p.name')]
    price_texts = [tag.text.strip() for tag in soup.select('span.price')]

    if not names:
        print(f"페이지 {page}에서 더 이상 상품을 찾을 수 없어 크롤링을 중단합니다.")
        break

    price_nums = []
    for txt in price_texts:
        num = re.sub(r'[^0-9]', '', txt)
        if num:
            price_nums.append(int(num))
    if len(price_nums) != len(names):
        print(f"경고: 페이지 {page}, 상품명과 가격 수가 달라요 ({len(names)}명 vs {len(price_nums)}개)")

    all_names.extend(names)
    all_prices.extend(price_nums)

    print(f"페이지 {page} 완료: 상품 {len(names)}개 수집됨.")
    page += 1

# 수집된 전체 결과 출력
print(f"\n총 수집된 상품 수: {len(all_names)}개")

if all_prices:
    arr = np.array(all_prices)
    print(f"평균 가격: {arr.mean():,.0f}원")
    print(f"가격 표준편차: {arr.std():,.0f}원")
    print(f"최고 가격: {arr.max():,}원")
    print("최고가 상품:")
    for name, price in zip(all_names, all_prices):
        if price == arr.max():
            print(f" - {name}")

