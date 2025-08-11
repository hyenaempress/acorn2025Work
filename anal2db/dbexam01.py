# pandas 문제 7)


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
    
    
#  a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.

#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성

    df1 = pd.DataFrame(cursor.fetchall(),
                       columns=['번호', '이름', '부서', '직급', '성별', '연봉'])
    print(df1.head(3))


#      - DataFrame의 자료를 파일로 저장

    df1.to_csv('jikwon.csv', encoding='utf-8', index=False)

#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력

    print(df1.groupby('부서')['연봉'].sum())
    print(df1.groupby('부서')['연봉'].max())
    print(df1.groupby('부서')['연봉'].min())


#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))

    ctab = pd.crosstab(df1['부서'], df1['직급'], margins=True)    

#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시

    print(df1.groupby('번호')['이름'].count())  # 직원별 담당 고객자료
    print(df1.groupby('번호')['이름'].count().map(lambda x: '담당 고객 X' if x == 0 else x))
   
#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성

    df1.groupby('부서')['연봉'].mean().plot(kind='bar', rot=0)
    plt.show()

#  b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.

#      - pivot_table을 사용하여 성별 연봉의 평균을 출력

    print(df1.pivot_table(index='성별', values='연봉', aggfunc='mean'))

#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프

    df1.pivot_table(index='성별', values='연봉', aggfunc='mean').plot(kind='bar', rot=0)
    plt.show()


#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))

    ctab = pd.crosstab(df1['부서'], df1['성별'], margins=True)
    print(ctab)



#  c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
#       조건 :  try ~ except MySQLdb.OperationalError as e:      사용
#      사번  직원명  부서명   직급  부서전화  성별
#      ...
#      인원수 : * 명
    
except Exception as e:
    print("SQL 실행 오류:", e)
finally:
    conn.close()

    

"""

    # 출력 1
    # for (a, b, c, d, e, f) in cursor:
    #     print(a, b, c, d, e, f)
    for(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay) in cursor:
        print(jikwonno, jikwonname, busername, jikwonjik, jikwongen, jikwonpay)

    # 출력 2: Dataframe
    df1 = pd.DataFrame(cursor.fetchall(),
                       columns=['jikwonno', 'jikwonname', 'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])
    print(df1.head(3))
    print()

    # 출력 3: csv 파일
    with open('./01_big_data_machine_learning/data/jik_data.csv', mode='w', encoding='utf-8') as fobj:
        writer = csv.writer(fobj)
        for r in cursor:
            writer.writerow(r)

    # csv 파일을 읽어 DataFrame에 저장
    df2 = pd.read_csv('./01_big_data_machine_learning/data/jik_data.csv', encoding='utf-8', names = ['번호', '이름', '부서', '직급', '성별', '연봉'])
    print(df2.head(3))

    print('\nDB에 자료를 pandas의 sql 처리 기능으로 읽기 ------')
    df = pd.read_sql(sql, conn)
    df.columns = ['번호', '이름', '부서', '직급', '성별', '연봉']
    print(df.head(3))

    print("\nDB의 자료를 DataFrame으로 읽었으므로 pandas의 기능을 적용 가능---")
    print('건수: ', len(df))
    print('건수: ', df['이름'].count())
    print('직급별 인원수', df['직급'].value_counts())
    print('연봉 평균: ', df.loc[:, '연봉'].mean())
    print()
    ctab = pd.crosstab(df['성별'], df['직급'], margins=True) # 성별 직급별 건수
    # print(ctab.to_html)

    # 시각화 - 직급별 연봉 평균 - pi
    jik_ypay = df.groupby(['직급'])['연봉'].mean()
    print('직급별 연봉 평균:\n', jik_ypay)

    print(jik_ypay.index)
    print(jik_ypay.values)
    plt.pie(jik_ypay, explode=(0.2, 0, 0, 0.3, 0), labels=jik_ypay.index, shadow=True, labeldistance=0.7, counterclock=False)
    plt.show()
"""
    
