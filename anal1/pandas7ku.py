# 웹 문서 읽기
import csv
import re
import pandas as pd
import urllib.request as req
from bs4 import BeautifulSoup
import urllib

import requests

# 위키백과 문서 읽기
# url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
# df = pd.read_html(url)

# wiki = req.urlopen(url)
# print(wiki)
# pip install beautifulsoup4
# soup = BeautifulSoup(wiki, "html.parser")
# print(soup.select("#mw-content-text > div.mw-parser-output > p"))

# 네이버 제공 코스피 정보 읽기 - DataFrame에 담아서 처리
url_template = "https://finance.naver.com/sise/sise_market_sum.naver"
csv_fname = '네이버코스피.csv'

with open(csv_fname, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    headers = ['No','종목명', '현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량', 'PER', 'ROE']
    writer.writerow(headers)

    for page in range(1, 2):  # 1페이지부터 2페이지까지
        url = url_template.format(page) 
        # print(url)
        res = requests.get(url)
        res.raise_for_status() # 실패하면 작업 정리
        soup = BeautifulSoup(res.text, "html.parser")

        # rows = soup.find('table', attrs={'class': 'type_2'}).find('tbody').find_all('tr')
        rows = soup.select('table.type_2 tbody tr')
        # print(rows)

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < len(headers):
                print(f"[스킵됨] 열 수 부족: {len(cols)} < {len(headers)}")
                continue
            row_data = [re.sub(r'[\n\t]+', '', col.get_text()) for col in cols]
            writer.writerow(row_data)
print('csv 저장 성공')