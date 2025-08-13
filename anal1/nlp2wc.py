#pip install pygame pytagcloud simplejson
#동아일보 검색 기능으로 문자열을 읽어 형태소 분석 후 워드 클라우드로 출력 
#https://www.donga.com/news/search?query=%EC%A4%91%EB%B3%B5


from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
from urllib.parse import quote

from pytagcloud import make_tags, create_tag_image, LAYOUT_HORIZONTAL


keyword = input('검색어를 입력하세요: ')
print('검색어: ', keyword)

target_url = 'https://www.donga.com/news/search?query=' + quote(keyword) #검색어를 인코딩 해줍니다.
#https://www.donga.com/news/search?query=무더위
print('target_url: ', target_url)

source_code = urllib.request.urlopen(target_url)
soup = BeautifulSoup(source_code, 'html.parser', from_encoding='utf-8')
#print(soup)

#contents > div > div > div.divide_area > section > div.sch_section.sch_news > ul > li:nth-child(1) > article > div > h4

mag = ""
for title in soup.find_all('h4', class_='tit'):
    title_link = title.select('a')
   # print(title_link)
    article_url = title_link[0]['href']
    try:
        source_article = urllib.request.urlopen(article_url)
        soup = BeautifulSoup(source_article, 'lxml.parser', from_encoding='utf-8')
        content = soup.select_one('div.article_txt').get_text()
        for imsi in content:
            item = str(imsi.find_all(text=True))
            #print(item)
            mag += item #msg에 차곡 차곡 쌓을게요 
        #print(content)
        print(mag)
    except Exception as e:
        pass
# print(mag)

from konlpy.tag import Okt
from collections import Counter
okt = Okt()

nouns = okt.nouns(mag)
print(nouns)

result = []
for imsi in nouns:
    if len(imsi) > 1:
        result.append(imsi)

print(result)



rh




