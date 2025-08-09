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

#한글 깨짐 방지 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import pickle
import sys


#연결 
"""
#마이에스큐엘 커넥트가 원하는 내용 
conn = MySQLdb.connect(host='127.0.0.1',
                       user='root', 
                       password='970506', 
                       db='mydb',
                       port=3306,
                       charset='utf8') #사실 이 내용은 다른곳에 있는게 좋겠죠? 그래서 피클을 씁니다다
                       


config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '970506',
    'db': 'mydb',
    'port': 3306,
    'charset': 'utf8'
} #이런식으로 딕트로 만들어서 넣어 줄 수 있습니다. 
"""

try:
    with open('mymaria.dat', mode= 'rb') as obj:
        config = pickle.load(obj)
    print(config)
except Exception as e:
    print('오류 발생 : ', e)
    sys.exit(0)

try:
 
    conn = MySQLdb.connect(**config) ##딕드라서 언팩 연산자를 써야 합니다. 언팩킹을 한것입니다. 딕셔너리 언팩킹 
   # sql = "select jikwonno, jikwonname from jikwon"
    #두번째 SQL 예제 
    sql = """
    select jikwonno, jikwonname, buser_num, jikwon_pay
    from jikwon
    where jikwonpay >= 1000000
    order by jikwonpay desc
       
    """
    #가독성이 좋을려면 for 방법이 좋습니다.
    
    cursor = conn.cursor()
    cursor.execute(sql) 
    #출력 1 
    for (a,b) in cursor:
        print(a,b)
        
    #출력 2
    df1 = pd.DataFrame(cursor.fetchall(),
                       columns=['jikwonno', 'jikwonname', 'buser_num', 'jikwonpay'])
    print(df1.head(3))    
    #출력 3
    df2 = pd.read_sql(sql, conn)
    print(df2.head(3))
    
except Exception as e:
    print('오류 발생 : ', e)
finally:
    cursor.close() #쓸모 없어짐 
    conn.close() 
    
