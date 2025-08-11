"""
a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
     - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
     - DataFrame의 자료를 파일로 저장
     - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
     - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
     - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
     - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
"""

import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.
import sys
import pickle
import csv

try:
    with open('myMaria.dat', mode='rb') as obj:
        config = pickle.load(obj)                   #피클스를 이용해 다른파일(myMaria.dat)에 미리 저장한 접속정보 부르기
except Exception as error:
    print('readError! : fail to read myMara', error)#접속정보 부르기에 실패했을 경우의 예외처리
    sys.exit()

try:
    #A1 : 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    connect = MySQLdb.connect(**config)     #피클스로 불러온 myMaria.dat의 내용물 언패킹(**)
    cursor = connect.cursor()
                                            #SQL 문법을 그대로 활용해서 jikwon 테이블에 buser 테이블은 이너조인해서 읽어오기
    sql = """
        SELECT jikwonno, jikwonname, busername, jikwonpay, jikwonjik
        FROM jikwon INNER JOIN buser
        ON jikwon.busernum = buser.buserno
    """
    cursor.execute(sql)                     #위의 SQL문 실행, cursor에 내용물 임시저장
    myDf = pd.DataFrame(cursor.fetchall(), columns=['사번', '이름', '부서명', '연봉', '직급'])  #SQL문으로 읽어온 내용을 모두 활용해 데이터프레임 만들기, 칼럼명 임의지정
    
    
    #A2 : DataFrame의 자료를 파일로 저장
    with open('IHateNamingFiles.csv', mode = 'w', encoding='utf-8') as obj: #쓰기모드(w), utf8 인코딩(한글문제)으로 새로운 CSV 파일 읽어오기
        writer = csv.writer(obj)
        for r in cursor:
            writer.writerow(r)
    
    
    #A3 : 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    print('부서명 별 연봉의 합 : ')
    filteredDf = myDf[myDf['부서명'] == '총무부']   #기존 데이터프레임에서 부서명이 총무부인 것들만 읽어온 새 데이터프레임 만들기
    print('총무부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())    #해당 데이터프레임에서 사용할 정보들만 읽어오기
    filteredDf = myDf[myDf['부서명'] == '영업부']
    print('영업부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '전산부']
    print('전산부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '관리부']
    print('관리부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    
    
    #A4 : 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    ctab = pd.crosstab(myDf['부서명'], myDf['직급'])        #ctab에 두 데이터프레임을 크로스테이블한 새 데이터프레임 만들기
    print('부서명, 직급으로 교차 테이블')
    print(ctab)                                            #해당 데이터프레임 그대로 출력
    
    
    #A5 : 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
    sql2 = """
        SELECT gogekno, gogekname, gogektel, gogekdamsano
        FROM gogek
    """
    cursor.execute(sql2)    #두 번째 SQL문법을 활용해 gogek 테이블에서 고객번호, 고객명, 고객전화번호, 고객담당자 사원번호 읽어와서 커서에 저장
    myDf2 = pd.DataFrame(cursor.fetchall(), columns=['고객번호', '고객명', '고객전화', '사번']) #위에서 읽어온 SQL문으로 새 데이터프레임 만들기
    myDf3 = pd.merge(myDf, myDf2, on='사번', how='outer')             #두개의 데이터프레임을 아우터조인으로 합치기
    myDf3['고객명'].fillna('담당 고객 X', inplace=True)                 #NaN으로 뜨는 값들 중 '고객명'에 한정해서 '담당 고객 X'로 뜨게하기
    print(myDf3[['이름', '고객번호', '고객명', '고객전화']])
    
    
    #A6 : 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    moneyData = myDf.groupby(['부서명'])['연봉'].mean()     #matplot 제작에 활용할 데이터를 미리 저장
    #print(moneyData)
    plt.figure(figsize=(8, 6))                              #그래프 창 크기
    plt.barh(moneyData.index, moneyData.values)             #가로 막대그래프, x축 데이터, y축 데이터
    plt.title('부서명별 연봉의 평균')                          #그래프 제목 지정
    plt.show()                                              #그래프 보이기

except Exception as error:
    print('failed to load MariaDB!', error)
finally:
    connect.close()         ##데이터베이스 작업이 끝나면 반드시 실행해야함.