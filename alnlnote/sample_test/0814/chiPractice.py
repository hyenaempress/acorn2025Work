import pandas as pd
import numpy as np
import scipy.stats as stats
import sys
import pickle
import MySQLdb

def myChecker(p_value, a_value):
    p_value = p_value
    significanceLevel = a_value
    
    if(p_value > significanceLevel):
        print(f"P값은 {p_value}, 유의수준 알파 {significanceLevel}보다 큼. 따라서 대립가설을 기각, 귀무가설을 채택함.")
    elif(p_value < significanceLevel):
        print(f"P값은 {p_value}, 유의수준 알파 {significanceLevel}보다 작음. 따라서 대립가설을 채택, 귀무가설을 기각함.")
    else:   # p_value == significanceLevel
        print("상식적으로 그럴 일 없음 ㅇㅇ")

    return 0


"""
카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오

  예제파일 : cleanDescriptive.csv
  칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
  조건 :  level, pass에 대해 NA가 있는 행은 제외한다.

  귀무가설 : 부모학력 수준이 자녀의 진학여부와 관련이 없다. (두 개는 서로 연관이 없음)
  대립가설 : 부모학력 수준이 자녀의 진학여부와 관련이 있다. (두 개는 서로 연관을 가짐)
"""
def chiPracticeOne():
    #데이터 가져오기
    data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv')
    data = data[['level2', 'pass2']]
    data = data.dropna()
    print(data)

    #빈도표
    ctab = pd.crosstab(index=data['level2'], columns=data['pass2'], margins=True)
    ctab.columns = ['실패', '합격', '행합']
    ctab.index = ['고졸', '대졸', '대학원졸', '열합']
    print(ctab)

    #p-value와 함수를 사용해서 가설검정
    chi2, p, dof, expected = stats.chi2_contingency(ctab)
    print(chi2, p, dof, expected)
    msg = "Test statistic : {}, p-value : {}"
    print(msg.format(chi2, p))
    #결론 : P값은 0.8374, 유의수준 알파 0.05보다 큼. 따라서 대립가설을 기각, 귀무가설을 채택함.
    myChecker(p, 0.05)
    #부모의 학력수준은 자녀의 진학 여부와 관련이 없다.
    print("부모의 학력수준은 자녀의 진학 여부와 관련이 없다.")

"""
카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.

  예제파일 : MariaDB의 jikwon table 
  jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
  jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
  조건 : NA가 있는 행은 제외한다.

  귀무가설 : 직급과 연봉은 관련이 없다. (독립이다, 연관성이 없다)
  대립가설 : 직급과 연봉은 관련이 있다. (독립이 아니다, 연관성이 있다)
"""
def chiPracticeTwo():
    #데이터 가져오기
    try:
        with open('myMaria.dat', mode='rb') as obj:
            config = pickle.load(obj)
    except Exception as error:
        print('DB 읽어오기 오류', error)
        sys.exit()

    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()
        sql = """
            SELECT jikwonjik, jikwonpay
            FROM jikwon
        """
        cursor.execute(sql)
        data = pd.DataFrame(cursor.fetchall(), columns=['직급', '연봉'])
        print(data)

        data.loc[data['직급'] == '이사', '직급'] = 1
        data.loc[data['직급'] == '부장', '직급'] = 2
        data.loc[data['직급'] == '과장', '직급'] = 3
        data.loc[data['직급'] == '대리', '직급'] = 4
        data.loc[data['직급'] == '사원', '직급'] = 5

        data.loc[data['연봉'] > 7000, '연봉'] = 4
        data.loc[data['연봉'] > 5000, '연봉'] = 3
        data.loc[data['연봉'] > 3000, '연봉'] = 2
        data.loc[data['연봉'] > 1000, '연봉'] = 1   #파이썬은 인터프리터이기에 이렇게 해도 가능, 다른언어에선 X
        print(data)
            
        #빈도표
        ctab = pd.crosstab(index=data['직급'], columns=data['연봉'])
        print(ctab)
        
        #p-value와 함수를 사용해서 가설검정
        chi2, p, dof, _ = stats.chi2_contingency(ctab)
        msg = ("Test statics : {}, P-value : {}, df : {}")
        print(msg.format(chi2, p, dof))

        #결론 : P값은 0.00029, 유의수준 알파 0.05보다 작음. 따라서 대립가설을 채택, 귀무가설을 기각함.
        myChecker(p, 0.05)
        #직급과 연봉은 관련이 있다.
        print("직급과 연봉은 관련이 있다.")

    except Exception as error:
        print('뭔진 모르겠지만 에러남, 확인해봐 ㅇㅇ', error)
    
    finally:
        cursor.close()
        conn.close()        #메모리 절약을 위한 종료들


"""
메인, 당장 확인할것만 주석 지우고 사용
"""
chiPracticeOne()
chiPracticeTwo()