
#날짜 데이터 
from bs4 import BeautifulSoup
import urllib.request  #연습용 실무용은 아님, 코드가 장황하다  
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
"""
url = "http://www.kma.go.kr/XML/weather/sfc_web_map.xml" #jsp 도 읽을 수 있습니다.
#data = urllib.request.urlopen(url).read() #유알엘 오픈 
#print(data)
#print(data.decode('utf-8')) #안보일떈 디코드 해주세요 


soup = BeautifulSoup(urllib.request.urlopen(url), 'xml') #lxml 이라고 하니까 경고가 떠서 변경 
#print(soup)

#이제 날씨 데이터를 데이터 프레임에 넣어 봅니다.

local = soup.find_all("local") 

df = pd.DataFrame(columns=["지역", "온도", "날씨"])

data = []
for loc in local:
    city = loc.text
    temp = loc.get("ta")
    data.append([city, temp])
    
df = pd.DataFrame(data, columns=["지역", "온도"])
print(df.head())
#데이터를 잘 받아왔으면 저장할 수 있습 
df.to_csv("weather.csv", index=False, encoding='utf-8')
#DB 로 저장할 떈 TO SQL 로 저장할 수 있습니다. 
"""


"""
다른 방법 응용 
for loc in local:
    df.loc[len(df)] = [
        loc.text,             # 지역 이름
        float(loc["ta"]),     # 현재 기온
        loc["desc"]           # 날씨 설명
    ]

print(df.head())
print("가장 더운 지역:")
print(df[df["온도"] == df["온도"].max()])

df = pd.DataFrame(columns=['지역', '온도', '날씨', '최저기온', '최고기온'])

"""


#이제 csv 파일을 읽어서 데이터 프레임에 넣어 봅니다.
#웹에서 데이터를 얻을 수 있고 시스템에서 얻을 수도 있습니다. 
df = pd.read_csv("weather.csv", encoding='utf-8')
print(df.head(2))
print(df[0:2])
print(df.tail(2))
print(df[-2:len(df)])
#위에것은 복습 
print(df.iloc[0]) #첫번째 행 
print(df.iloc[0:2]) #첫번째 두번째 행
print(df.iloc[0:2, :]) #첫번째 두번째 행 모든 열
#iloc 로 운영할수 있고 lic 를 써서 운영할 수도 있고 이 둘의 차이는 뭐지? 
print()
print(df.loc[1:3,['온도']]) #1번째 2번째 3번째 행의 온도 열 
print(df.info()) #데이터 프레임의 정보를 가져온다 

#치킨 실습 때 처럼 평균 구할 수 있습니다.
print(df['온도'].mean()) #평균 구하기 
print(df['온도'].std()) #표준편차 구하기 
print(df['온도'].min()) #최소값 구하기 
print(df['온도'].max()) #최대값 구하기 

print(df.loc[df['온도'] >=20])   #온도가 20도 이상인 행을 출력 
print(df.sort_values(by='온도', ascending=False).head(3)) #온도 내림차순 정렬 후 상위 3개 행을 출력 
#어센딩으로 오름차순 정렬 할 수 있습니다. 

#데이터 프레임 정렬 
print(df.sort_values(by='온도', ascending=False)) #온도 내림차순 정렬 
print(df.sort_values(by='온도', ascending=True)) #온도 오름차순 정렬 







