# 동아일보 검색 기능으로 문자열을 읽어 형태소 분석 후 워드클라우드로 출력

from bs4 import BeautifulSoup   # HTML 파싱을 위한 라이브러리 임포트
import urllib.request           # 웹 페이지 요청을 위한 라이브러리 임포트
from urllib.parse import quote  # URL 인코딩을 위한 함수 임포트
from konlpy.tag import Okt      # 형태소 분석기 Okt 임포트
from collections import Counter # 단어 빈도 계산을 위한 Counter 임포트
import pytagcloud               # 워드클라우드 생성을 위한 라이브러리 임포트
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

keyword = "무더위"              # 검색할 키워드 지정
print(quote(keyword))           # 키워드를 URL 인코딩하여 출력

target_url = "https://www.donga.com/news/search?query=" + quote(keyword)  # 검색 결과 페이지 URL 생성
# target_url = "https://www.donga.com/news/search?query=" + keyword
# print(target_url)

source_code = urllib.request.urlopen(target_url)  # 검색 결과 페이지 요청
soup = BeautifulSoup(source_code, 'lxml', from_encoding='utf-8')  # HTML 파싱
# print(soup)

msg = ""  # 기사 본문을 저장할 변수 초기화
for title in soup.find_all('h4', class_='tit'):  # 기사 제목 영역 모두 찾기
    title_link = title.select('a')               # 제목 내 a 태그(링크) 추출
    print(title_link)                            # 링크 정보 출력
    article_url = title_link[0]['href']          # 기사 상세 페이지 URL 추출
    # article_url = title_link['href'] 0번째를 빼고 하면 안 돌아간다.
    print(article_url)                           # 기사 URL 출력
    try:
        source_article = urllib.request.urlopen(article_url)  # 기사 상세 페이지 요청
        soup = BeautifulSoup(source_article, 'lxml', from_encoding='utf-8')  # HTML 파싱
        contents = soup.select('div.article_txt')             # 기사 본문 영역 추출
        print(contents)                                       # 본문 내용 출력
        for temp in contents:
            item = str(temp.find_all(string=True))            # 본문 텍스트 추출
            print(item)                                       # 추출된 텍스트 출력
            msg += item                                       # 본문 텍스트 누적

    except Exception as e:
        pass  # 에러 발생 시 무시

# print(msg)

okt = Okt()                       # Okt 형태소 분석기 객체 생성
nouns = okt.nouns(msg)            # 본문에서 명사만 추출
print(nouns)                      # 추출된 명사 출력

result = []                       # 2글자 이상 명사만 저장할 리스트
for temp in nouns:
    if len(temp) > 1:             # 2글자 이상인 경우만
        result.append(temp) 
# print(result[:10])

count = Counter(result)           # 명사 빈도수 계산
# print(count)

tag = count.most_common(50)       # 빈도수 상위 50개 추출

taglist = pytagcloud.make_tags(tag, maxsize=100)  # 워드클라우드 태그 리스트 생성
print(taglist)                                 # 태그 리스트 출력

pytagcloud.create_tag_image(taglist,           # 워드클라우드 이미지 생성
                            './01_big_data_machine_learning/data/wordcloud.png',
                            size=(1000, 600), 
                            background=(0,0,0),
                            fontname = "korean",
                            rectangular=False)

# 이미지 읽기
img = mpimg.imread('./01_big_data_machine_learning/data/wordcloud.png')
plt.imshow(img)
plt.show()


