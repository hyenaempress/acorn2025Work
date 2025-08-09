#마리아 디비를 쓰는 이유는 마리아 디비가 SQL 표준을 따르고 있기 떄문입니다.
#이번에는 마리아 디비와 연동하는 방법을 알아보겠습니다.
#Postgres 디비도 많이 쓰입니다. 나중에 그것도 설치해서 연결해보는것도 좋습니다.
#모두 관계형 데이터 베이스입니다.


#1 연결할려면 커넥션 객체를 만들어야 하고 우선 모듈 설치부터 파겠습니다.
#아나콘다 프롬프트에서 pip install MySQLClient 을 해줍니다.
#이후엔 두개의 폴더가 생깁니다. 


import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import csv

plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

try:
    with open('./mymaria.dat', 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print("읽기 오류:", e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = "select jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay from jikwon inner join buser on jikwon.busernum=buser.buserno"

    cursor.execute(sql)

    # 출력 1
    # for (a, b, c, d, e, f) in cursor:
    #     print(a, b, c, d, e, f)
    for(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay) in cursor:
        print(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay)

    # 출력 2
    
    df1 = pd.DataFrame(cursor.fetchall(),
                       columns=['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])
    print(df1.head(3))    
    #출력 3
    df12 = pd.read_sql(sql, conn)
    print(df12.head(3))
    
    #CSV 파일로 출력력
    with open('jik_data.csv', 'w', encoding='utf-8') as fobj:#위드문 써야 클로우즈를 안합니다.
        writer = csv.writer(fobj)      
    
    #CSV 파일로 저장 
    df1.to_csv('jikwon.csv', encoding='utf-8', index=False)
    
    #CSV 핑;ㄹ러 
    df2 = pd.read_csv('jik_data.csv', encoding='utf-8',
                      names=['번호','이름','부서','직급','성별','연봉'])
    print(df2.head(3)) 
    #웹으로 출력 장고를 사용해야 합니다.
    print('\nDB의 자료를 pandasd의 sql 로처리')
    df = pd.read_sql(sql, conn)
    print(df.head(3))
    
    print('\n디비의 자료를 데이터 프레임으로 읽었음으로 판다스의 기능을 사용 할 수 있습니다')
    #판다스 기능 몇개만 적용해봅시다.
    print('건수 : ', len(df))
    print('평균 연봉 : ', df['jikwonpay'].mean())
    print('최고 연봉 : ', df['jikwonpay'].max())
    print('최저 연봉 : ', df['jikwonpay'].min())
    print('평균 연봉 : ', df['jikwonpay'].mean())
    print('최고 연봉 : ', df['jikwonpay'].max())
    print('최저 연봉 : ', df['jikwonpay'].min())
    #직급별 인원수 
    print(df['jikwonjik'].value_counts())
    #부서별 인원수 
    print(df['busername'].value_counts())
    #성별 인원수 
    print(df['jikwongen'].value_counts())
    #연봉 평균 
    print(df['jikwonpay'].mean())
    #연봉 최고 
    
    #연봉이 있으면 분산 표준 편차 쓸 수 있죠 판다스 쓰면 되겠죠 
    print('분산 : ', df['jikwonpay'].var())
    print('표준편차 : ', df['jikwonpay'].std())
    #연봉이 있으면 히스토그램 그려보죠 
    plt.hist(df['jikwonpay'], bins=10, color='green', alpha=0.5)
    plt.show()
    
   
    ctab = pd.crosstab(df['jikwongen'], df['jikwonjik'], margins=True) #성별 직급 별 건수
    print(ctab)
    
    #print(ctab.to_html())  # 이렇게 HTML 파일로 할 수 있다.
    
    #시각화 - 직급별 연봉 평균 
    df.groupby('jikwonjik')['jikwonpay'].mean().plot(kind='bar', rot=0)
    plt.show()
    print('직급별 연봉 평균 : ')
    
    #시각화 - 부서별 연봉 평균 
    df.groupby('busername')['jikwonpay'].mean().plot(kind='bar', rot=0)
    plt.show()
    #시각화 - 부서별 연봉 평균 pie 그래프
    df.groupby('busername')['jikwonpay'].mean().plot(kind='pie', autopct='%.1f%%')
    plt.show()
    
    plt.pie(df['jikwonpay'], labels=df['jikwonjik'], autopct='%.1f%%', explode=(0.1,0.1,0.1,0.1,0.1), shadow=True, startangle=90)
    plt.show()
    
    #시계 방향으로  그려보죠 
    plt.pie(df['jikwonpay'], labels=df['jikwonjik'], autopct='%.1f%%', explode=(0.1,0.1,0.1,0.1,0.1), shadow=True, startangle=90)
    plt.show()

except Exception as e:
    print("SQL 실행 오류:", e)

