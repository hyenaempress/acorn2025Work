from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 셀레니움을 사용해야 하는 이유
# - 봉구스밥버거의 메뉴 정보는 동적으로 로딩되는 웹 페이지로, 일반적인 HTTP 요청으로는 HTML 소스를 가져올 수 없습니다.
# - 따라서, 셀레니움을 사용하여 실제 브라우저를 통해 페이지를 로드하고, 필요한 데이터를 추출해야 합니다.

matplotlib.rc('font', family='Malgun Gothic')  # 윈도우: 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 깨짐 방지

# 0. 필요사항
# - 로컬에 Python 3.13.5 설치
# - Selenium 및 BeautifulSoup4, pandas, matplotlib, seaborn 설치
# -- pip install selenium beautifulsoup4 pandas matplotlib seaborn
# - https://googlechromelabs.github.io/chrome-for-testing/#stable 에서 크롬 드라이버 다운로드 및 경로 설정
# - (터미널) 아나콘다 자동 활성화 비활성화 명령어: conda config --set auto_activate_base false

# 1. 크롬 드라이버 경로 지정 (chromedriver.exe 위치에 맞게 수정)
chrome_path = "./01_big_data_machine_learning/code/0807/연습문제/chromedriver.exe"
service = Service(chrome_path)
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 창 안 띄우고 실행

# 2. 웹 페이지 열기
driver = webdriver.Chrome(service=service, options=options)
driver.get("https://bongousse.com/Menu_list.asp")
time.sleep(2)  # 페이지 로딩 대기

# 3. HTML 소스 가져오기
html = driver.page_source

# 4. BeautifulSoup 객체 생성 후 soup에 저장
soup = BeautifulSoup(html, "html.parser")

# 5. (로그) soup 파일의 내용을 log_downloaded_menu.html로 저장
# 로그 저장 활성화 여부
SAVE_LOG = False  # False로 바꾸면 로그 저장 안 함
if SAVE_LOG:
    with open("./01_big_data_machine_learning/code/0807/연습문제/log_downloaded_menu.html", "w", encoding="utf-8") as f:
        f.write(str(soup.prettify()))

# 6. 드라이버 종료
driver.quit()

# 7. soup 객체에서 메뉴명과 가격 추출
names = [tag.text.strip() for tag in soup.select('ul#bonmenu_list li div.txtbox > div.tit')]
prices = [int(tag.text.strip().replace(',', '')) for tag in soup.select('ul#bonmenu_list li div.txtbox > div.price')]

# 8. 읽어온 거 출력
PRINT_MENU_PRICE = False # False로 바꾸면 출력 안 함
if PRINT_MENU_PRICE:
    print("메뉴명:", names)
    print("가격:", prices)

# 9. 읽어온 데이터로 DataFrame 생성
# 메뉴명    가격    설명
df = pd.DataFrame({
    '상품명': names,
    '가격': prices,
    '설명': ['저는 봉구킹을 좋아합니다!'] * len(names)
})

PRINT_DATAFRAME = False # False로 바꾸면 출력 안 함
if PRINT_DATAFRAME:
    print(df)

# 건수:
print('메뉴 개수: \t', len(names), '개')
# 가격평균:
print('가격 평균: \t', round(df['가격'].mean(), 2), '원')
# 표준편차:
print('표준 편차: \t', round(df['가격'].std(), 2), '원')
# 최고가격:
print('최고 가격: \t', df['가격'].max(), '원')
# 최저가격:
print('최저 가격: \t', df['가격'].min(), '원')

# # 시각화
plt.figure(figsize=(16, 9))

# 1. 선 그래프 (가격 추이)
plt.subplot(3, 2, 1)
plt.plot(range(len(df)), df['가격'], 'r-o')
plt.title('메뉴별 가격 추이 (선 그래프)')
plt.xlabel('순서')
plt.ylabel('가격')
plt.grid()

# 2. 막대 그래프 (가격)
plt.subplot(3, 2, 2)
plt.bar(range(len(df)), df['가격'], color='skyblue')
plt.title('메뉴별 가격 (막대 그래프)')
plt.xlabel('순서')
plt.ylabel('가격')
plt.grid()

# 3. 히스토그램 (가격 분포)
plt.subplot(3, 2, 3)
plt.hist(df['가격'], bins=10, color='orange', edgecolor='black')
plt.title('가격 분포 (히스토그램)')
plt.xlabel('가격')
plt.ylabel('빈도')
plt.grid()

# 4. 박스플롯 (이상치 포함 가격 분포)
plt.subplot(3, 2, 4)
plt.boxplot(df['가격'], vert=False)
plt.title('가격 분포 (박스플롯)')
plt.xlabel('가격')
plt.grid()

# 5. 파이 차트 (상위 5개 메뉴 가격 비중)
top5 = df.nlargest(5, '가격')
plt.subplot(3, 2, 5)
plt.pie(top5['가격'], labels=list(top5['상품명']), autopct='%1.1f%%', startangle=140)
plt.title('상위 5개 메뉴 가격 비중 (파이 차트)')

# 6. 산점도 (순서 vs 가격)
plt.subplot(3, 2, 6)
plt.scatter(range(len(df)), df['가격'], c=df['가격'], cmap='viridis', s=100)
plt.title('순서 vs 가격 (산점도)')
plt.xlabel('순서')
plt.ylabel('가격')
plt.grid()

plt.tight_layout()
plt.show()

#seaborn 스타일 적용

# 스타일 적용 (seaborn 스타일)
sns.set_theme(style="whitegrid", font="Malgun Gothic", rc={"axes.unicode_minus":False})

plt.figure(figsize=(16, 12))

# 1. 선 그래프 (가격 추이)
plt.subplot(3, 2, 1)
sns.lineplot(x=range(len(df)), y=df['가격'], marker='o', color='crimson')
plt.title('메뉴별 가격 추이 (선 그래프)')
plt.xlabel('순서')
plt.ylabel('가격')

# 2. 막대 그래프 (가격)
plt.subplot(3, 2, 2)
sns.barplot(x=range(len(df)), y=df['가격'], palette='Blues_d')
plt.title('메뉴별 가격 (막대 그래프)')
plt.xlabel('순서')
plt.ylabel('가격')
plt.xticks(ticks=range(0, len(df), max(1, len(df)//10)), rotation=45, ha='right')  # x축 레이블을 10개 간격으로만 표시

# 3. 히스토그램 (가격 분포)
plt.subplot(3, 2, 3)
sns.histplot(data = df, x='가격', bins=10, color='orange', edgecolor='black')
plt.title('가격 분포 (히스토그램)')
plt.xlabel('가격')
plt.ylabel('빈도')

# 4. 박스플롯 (이상치 포함 가격 분포)
plt.subplot(3, 2, 4)
sns.boxplot(x=df['가격'], color='limegreen')
plt.title('가격 분포 (박스플롯)')
plt.xlabel('가격')

# 5. 파이 차트 (상위 5개 메뉴 가격 비중) - matplotlib 사용
top5 = df.nlargest(5, '가격')
plt.subplot(3, 2, 5)
plt.pie(top5['가격'], labels=list(top5['상품명']), autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('상위 5개 메뉴 가격 비중 (파이 차트)')

# 6. 산점도 (순서 vs 가격)
plt.subplot(3, 2, 6)
sns.scatterplot(x=range(len(df)), y=df['가격'], hue=df['가격'], palette='viridis', s=100, legend=False)
plt.title('순서 vs 가격 (산점도)')
plt.xlabel('순서')
plt.ylabel('가격')

plt.tight_layout(pad=3.0)  # pad 값을 늘려 간격 확보
plt.show()