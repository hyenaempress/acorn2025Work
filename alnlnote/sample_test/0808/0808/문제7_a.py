
# README : 위 코드는 정상적으로 동작하긴 하지만, 이걸 짠 내가 생각해도 공간복잡도가 좋지 못함
# 또한, 주석 역시 여전히 부족함.
# 개선된 버전은 주말에 다시 올릴 예정

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
        config = pickle.load(obj)
except Exception as error:
    print('readError! : fail to read myMara', error)
    sys.exit()

try:
    #A1 : 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    connect = MySQLdb.connect(**config)
    cursor = connect.cursor()

    sql = """
        SELECT jikwonno, jikwonname, busername, jikwonpay, jikwonjik
        FROM jikwon INNER JOIN buser
        ON jikwon.busernum = buser.buserno
    """
    cursor.execute(sql)
    myDf = pd.DataFrame(cursor.fetchall(), columns=['사번', '이름', '부서명', '연봉', '직급'])
    #A2 : DataFrame의 자료를 파일로 저장
    with open('IHateNamingFiles.csv', mode = 'w', encoding='utf-8') as obj:
        writer = csv.writer(obj)
        for r in cursor:
            writer.writerow(r)
    #A3 : 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    print('부서명 별 연봉의 합 : ')
    filteredDf = myDf[myDf['부서명'] == '총무부']
    print('총무부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '영업부']
    print('영업부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '전산부']
    print('전산부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '관리부']
    print('관리부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    #A4 : 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    ctab = pd.crosstab(myDf['부서명'], myDf['직급'])
    print('부서명, 직급으로 교차 테이블')
    print(ctab)
    #A5 : 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
    sql2 = """
        SELECT gogekno, gogekname, gogektel, gogekdamsano
        FROM gogek
    """
    cursor.execute(sql2)
    myDf2 = pd.DataFrame(cursor.fetchall(), columns=['고객번호', '고객명', '고객전화', '사번'])
    myDf3 = pd.merge(myDf, myDf2, on='사번', how='outer')
    myDf3['고객명'].fillna('담당 고객 X', inplace=True)
    print(myDf3[['이름', '고객번호', '고객명', '고객전화']])
    #A6 : 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    moneyData = myDf.groupby(['부서명'])['연봉'].mean()
    #print(moneyData)
    plt.figure(figsize=(8, 6))
    plt.barh(moneyData.index, moneyData.values)
    plt.title('부서명별 연봉의 평균')
    plt.show()

except Exception as error:
    print('failed to load MariaDB!', error)
finally:
    connect.close()