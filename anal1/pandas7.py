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

"""
# 네이버 제공 코스피 정보 읽기 - DataFrame에 담아서 처리
url_template = "https://finance.naver.com/sise/sise_market_sum.naver"
csv_fname = '네이버코스피.csv'

with open(csv_fname, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    headers = ['종목명', '현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량', 'PER', 'ROE']
    writer.writerow(headers) #스플릿을 넣을 수도 있다 예) 종목명,현재가,전일비,등락률,액면가,시가총액,상장주식수,외국인비율,거래량,PER,ROE


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
"""
#이제 읽어옵니다
df = pd.read_csv('네이버코스피.csv', dtype=str, index_col=False)
print(df.head(3)) # 3개만 출력

# 종목명 컬럼 추출
df['종목명'] = df['종목명'].str.replace('\n', '')
print(df.head(3))

# 현재가 컬럼 추출
df['현재가'] = df['현재가'].str.replace(',', '')
print(df.head(3))
print(df.info()) # 데이터 타입 확인

#숫자형으로 반환 할 칼럼

numeruc_cols = ['현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '거래량', 'PER', 'ROE']

#전일비 전용 전처리 함 수 
def clean_change_direction(val):
    if pd.isna(val):    
        return pd.NA
    val = str(val)
    val = val.replace(',', '').replace('상승', '+').replace('하락', '-') #콤마 제거 상승 하락 기호 추가
    val = re.sub(r'[^\d\.\-\+]', '', val) #숫자와 기호 제외 제거
    try:
        return float(val)
    except ValueError:
        return pd.NA #값이 없을시 NA 반환
    
#전일비 컬럼 전처리
df['전일비'] = df['전일비'].apply(clean_change_direction)
print(df.head(3))

#일반 숫자형 컬럼 전처리 
def clean_numeric_column(series):
    return ( series.astype(str).str
            .replace(',', '', regex=False) #콤마 제거
            .replace('%', '', regex=False) #% 제거
            .replace(['','-','N/A','nan'], pd.NA)   #빈 값, -, N/A, nan 을 NA 로 변환
            .apply(lambda x:pd.to_numeric(x, errors='coerce')) #숫자로 변환 실패시 NA 로 변환
            )
    
for col in numeruc_cols:
    df[col] = clean_numeric_column(df[col])
print('숫자 컬럼 일괄처리 완료')
print(df.head(3))
print(df[['종목명','현재가','전일비']].head(3))
#시가총액 top 5

print('시가총액 top 5')
top_5 = df.sort_values(by='시가총액', ascending=False).head(5)
top_5_1 = df.dropna(subset=['시가총액']).sort_values(by='시가총액', ascending=False).head(5)
#
print(top_5)
print(top_5_1)

#코스피 데이터를 읽어서 데이터 프레임을 만들 수 있다. 판다스의 기능을 이용을 할 수 있다. 
#판다스의 다양한 기능을 웹에서 쉽게 사용할 수 있다.   