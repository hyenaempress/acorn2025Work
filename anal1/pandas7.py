#웹문서 읽기 
#위키백과 문서 읽기 - 이순신 자료 
import urllib.request as req
from bs4 import BeautifulSoup
# pip install beautifulsoup4
import urllib

"""""
url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
#한글을 넣고 싶으면 파싱 해야한다
wiki = req.urlopen(url)
print(wiki)

soup = BeautifulSoup(wiki, 'html.parser') 
print(soup.select("#mw-content-text> div.mw-parser-output>p")) #이거 말고 find 쓸 수도 있음 
"""

#네이버 제공 코스피 정보 읽기 - DataFrame에 담아 어쩌구..

url_tem = "https://finance.naver.com/sise/nxt_sise_market_sum.naver"
csv_fname = '네이버코스피'