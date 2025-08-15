# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
# 그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
#   예제파일 : MariaDB의 jikwon table 
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#   조건 : NA가 있는 행은 제외한다.

# 대립가설(H0): A회사의 직급과 연봉은 관련이 있다.
# 귀무가설(H1): A회사의 직급과 연봉은 관련이 없다. 

import pandas as pd
import scipy.stats as stats
import MySQLdb
import pickle
import sys

try:
    with open('./01_big_data_machine_learning/data/mymaria.dat', 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print("읽기 오류:", e)
    sys.exit()

try:
    # 데이터 읽어오기 
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = "select jikwonjik, jikwonpay from jikwon"

    cursor.execute(sql)

    # 데이터 프레임 생성
    df = pd.DataFrame(cursor.fetchall(),
                    columns=['jikwonjik', 'jikwonpay'])
    print(df.head(3))

    # 직급 매핑
    jik_map = {'이사': 1, '부장': 2, '과장': 3, '대리': 4, '사원': 5}
    df['jikwonjik'] = df['jikwonjik'].replace(jik_map)

    # 연봉 매핑
    bins = [1000, 3000, 5000, 7000, float('inf')]
    labels = ["1", "2", "3", "4"]
    df['jikwonpay'] = pd.cut(df['jikwonpay'], bins=bins, labels=labels, right=False)

    # 크로스테이블 생성 (매핑된 값 기준)
    ctab = pd.crosstab(index=df['jikwonjik'], columns=df['jikwonpay'])
    print(ctab)

    # 함수 사용 - p-value 판정
    chi2, p, dof, expected = stats.chi2_contingency(ctab)
    # 판정
    if p <= 0.05:  # type: ignore
        print('기각')
    else:
        print('채택')

    print(chi2, p, dof, expected)
    print("자유도: ", dof)
    # Test statistic: 37.40, p-value: 0.0002
    # 메시지 출력
    msg = "Test statistic: {:.2f}, p-value: {:.4f}".format(chi2, p)
    print(msg)  # 여기서 format() 다시 쓰면 안 됨


    # 결론: p-value(0.0002) < 유의수준(0.05) 이므로 귀무가설을 기각한다.
    # 따라서, A회사의 직급과 연봉은 관련이 있다.

except Exception as e:
    print("SQL 실행 오류:", e)
finally:
    conn.close()