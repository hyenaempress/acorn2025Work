
#웹 스크래핑

from bs4 import BeautifulSoup
import urllib.request  #연습용 실무용은 아님, 코드가 장황하다  
import requests         #실전용에 가까움 , 코드가 비교적 간결하다 
import pandas as pd
import numpy as np

url = "https://www.kyochon.com/menu/chicken.asp"
#asp는 닷넷입니다. 참고로 알아두세요 C# 사용자도 있고 베이직도 있고...
response = requests.get(url) #get 요청을 보내고 응답을 받는다. 
response.raise_for_status() # 응답 코드가 200이 아니면 예외를 발생시킨다. 웹에서 꼭 중요함  이거를 for문 써서 처리하지만 너무 자주하면 차단당함



#셀/7 셀레니움 크롤림 스래피 이런걸 가지고 데이터를 긁어야 좀더 전문적이다. 
#이걸 보여주고 싶은데 노가다를 좀 만많이 해야합니다.  
#교촌치킨에서 상품들에 대한 가격의 평균 표준편차 4분위수 상품에 대해서 여러가지 작업들을 해낼 수 있다. 
#판다스 어느정도 이야기 하고 맥플로립으로 넘어갈 수 있도록 합니다. 
#수업시간에 문제내고 있지만 그것보다 쉽습니다.
#판판
#아직 트라이 익셉트 안썼는데 네트워크는 트라이 익셉트 해주세요

soup = BeautifulSoup(response.text, 'html.parser') #lxml 이라고 해도 됨 
#print(soup) 긁어온거 확인 
#매뉴 이름 추출 
names = [tag.text.strip() for tag in soup.select('dl.txt>dt')] #find_all 이라고 해도 됨 
#print(names)

#가격 이름 추출
prices = [int(tag.text.strip().replace(',', '')) for tag in soup.select('p.money strong')]
#print(prices)

df = pd.DataFrame({'상품명': names, '가격': prices})
print(df.head(3))
print('가격 평균:', round(df['가격'].mean(), 2)) #라운드를 주는 이유  
print('가격 표준편차:', round(df['가격'].std(), 2)) #라운드를 써도 되고 
print(f"가격평균:{df['가격'].mean():.2f}, 표준편차:{df['가격'].std():.2f}") #f스트링 써도 되고 그럼 플롯으로 출력됩니다 
#XML 을 쓸 수도 있다. 그런 데이터를 안준다면 직접 서버를 만들어서 써도 된다. 장고서버 같은걸로 써도 된다. 
#로컬로 아니고 고정아이피를 쓴다고 하면 웹서버를 만들어서 써도 된다. 
#https://github.com/pykwon/python



#이번엔 XML 데이터를 불러와서 작업해보자 , 어디가면 있냐 깃허브에 가면 있다.


