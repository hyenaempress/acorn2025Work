import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import csv
from pandas import DataFrame, Series

# pandas 문제 7)
# b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))


plt.rc('font', family= 'malgun gothic')     # 한글 깨짐 방지 코드 두줄
plt.rcParams['axes.unicode_minus']= False   # 한글 깨짐 방지 코드 두줄

config = {
    'host': '127.0.0.1',
    'user' : 'root',
    'password' : 'skfrnwl1@',
    'database' : 'mydb',
    'port' : 3306,
    'charset' : 'utf8'
}

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    
    sql = '''
        select jikwonname,busername, jikwonjik, jikwongen, jikwonpay
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
            '''
    cursor.execute(sql)
    df2 = pd.DataFrame(cursor.fetchall(),
                      columns = ['jikwonname','busername',
                                 'jikwonjik','jikwongen','jikwonypay']
                      )

#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
    # pivot_table(테이블을 생성할 원본, index = 행으로 사용할 열들의 이름, cloumns = 열로 사용할 열들의 이름, values = 집계 대상 열 이름)
    df2.pivot_table(index = 'jikwongen', values = 'jikwonypay')

    # 실행할때 주석 해제
    # print(df2)

    man = df2[df2['jikwongen'] == '남']
    woman = df2[df2['jikwongen'] == '여']

    mean_manYpay = round(man['jikwonypay'].mean(),2)
    mean_womanYpay = round(woman['jikwonypay'].mean(),2)
    print('남성 직원 평균 연봉 : ', mean_manYpay)
    print('여성 직원 평균 연봉 : ', mean_womanYpay)
    
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프

    # bar(x좌표 내용, y좌표 내용 - 변수 또는 문자열 사용 가능) 클래스 사용
    labels = ['남성 직원 평균 연봉', '여성 직원 평균 연봉']
    values = [mean_manYpay,mean_womanYpay]
    plt.bar(labels, values)
    plt.title(config['database'])
    plt.ylabel('단위 : (만) 원')

    # 실행할때 주석 해제
    # plt.show()

#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
    # 변수의 빈도를 계산해줌
    # crosstab(index - 행으로 쓸 변수, columns - 열로 쓸 변수)
    # 각각의 busername에 jikwongen의 요소값(성별)이 몇번 나오는지 출력해줌
    ctab2 = pd.crosstab(df2['busername'],df2['jikwongen'])
    print(ctab2)
    print(df2.head(3))

except Exception as e:
    print('처리 오류 : ', e)
finally:
    conn.close()