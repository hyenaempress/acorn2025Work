import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import levene
import random
import MySQLdb
import matplotlib.pyplot as plt
import sys
import pickle
import csv
# [two-sample t 검정 : 문제1] 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
#    blue : 70 68 82 78 72 68 67 68 88 60 80
#    red : 60 65 55 58 67 59 61 68 77 66 66

# 귀무가설(H0) : 포장지 색상에 따른 매출액은 변동이 없다.
# 대립가설(H1) : 포장지 색상과 매출액은 관련이 있다.

blue = [70,68,82,78,81,68,67,68,88,60,80]
red = [60,65,55,58,67,59,61,68,77,66,66]

leven_stat, leven_p = levene(blue, red)
print('포장지 색 leven_stats : ',leven_stat, ' ', 'leven_p : ',leven_p)
# leven_stats :  1.4116963594113088   leven_p :  0.2486933911637059
# p값이 유의수준 0.05 이상이다 따라서 등분산성 성립

two_sample = stats.ttest_ind(blue,red,equal_var = True)
print(two_sample)
# TtestResult(statistic=np.float64(3.1073508233015548),         이건 등분산성 성립 검정 후의 결과
# pvalue=np.float64(0.005551539548670424), df=np.float64(20.0))

# TtestResult(statistic=np.float64(3.1073508233015548),         이건 등분산성 성립 검정 전의 결과
# pvalue=np.float64(0.005551539548670424), df=np.float64(20.0))

# t-test의 결과 p-value값이 0.05 이하이므로 귀무가설(H0)을 기각한다.
# 결론 : 포장지 색상과 매출액은 관련이 있다.


# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
#   남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
#   여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4

# 귀무가설(H0) : 혈관 내의 콜레스테롤 양은 남녀 성별에 관계가 없다.
# 대립가설(H1) : 혈관 내의 골레스테롤 양은 남녀 성별에 관계가 있다.

man = [0.9,2.2,1.6,2.8,4.2,3.7,2.6,2.9,3.3,1.2,3.2,2.7,3.8,4.5,4,2.2,0.8,0.5,0.3,5.3,5.7,2.3,9.8]
woman = [1.4,2.7,2.1,1.8,3.3,3.2,1.6,1.9,2.3,2.85,2.3,1.4,2.6,3.5,2.1,6.6,7.7,8.8,6.6,6.4]
# print(len(man),len(woman))

sample_man = random.sample(man, 15)
sample_woman = random.sample(woman, 15)
leven_stat, leven_p = levene(sample_man, sample_woman)
print('콜레스테롤 leven_stat : ',leven_stat, ' ', 'p-value : ', leven_p)
# 콜레스테롤 leven_stat :  0.5222067169956457   p-value :  0.4758949224811917 
# p-value가 0.05 이상이므로 남녀간 데이터에는 등분산성이 성립 -> 독립 표본 검정에 사용
two_sample = stats.ttest_ind(sample_man, sample_woman, equal_var = True)
print('콜레스테롤 Two Sample : ',two_sample)
# 콜레스테롤 Two Sample :  TtestResult(statistic=np.float64(-0.6529739708148931), 
# pvalue=np.float64(0.5190990694959919), df=np.float64(28.0))
# p-value가 유의수준 0.05 이상이므로 귀무가설을 채택한다.
# 콜레스테롤 수치에서의 성별 여부는 관계가 없다.
 

# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다. replace 사용? 근데 필요 없어보여

# 귀무가설(H0) : 총무부와 영업부 직원들 사이의 부서에 따른 연봉 격차는 없다.
# 대립가설(H1) : 총무부와 영업부 직원들의 연봉은 부서에 영향을 받았다.

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = """
        select jikwonpay as 연봉, busernum as 부서번호, busername as 부서명
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
        """
    cursor.execute(sql)
    df = pd.DataFrame(cursor.fetchall(),
                      columns = ['연봉', '부서번호','부서명']
                      )
    print(df)

    # sp = np.array(data.iloc[:, [1,4]])  # AMT, rain_yn
    payment = df[['연봉','부서명']]
    print(payment)
    chong = payment[payment['부서명'] == '총무부']
    print('총무부 : ',chong)
    young = payment[payment['부서명'] == '영업부']
    print('영업부 : ', young)
    avg_ch = chong['연봉'].mean()
    avg_y = young['연봉'].mean()
    chong = pd.to_numeric(chong['연봉'])
    young = pd.to_numeric(young['연봉'])

    # 등분산성 먼저 검정
    leven_stat, leven_p = levene(young,chong)
    print('영업부와 총무부의 등분산성 검정 -> leven_stat : ',leven_stat, 'leven_p : ',leven_p)
    # 영업부와 총무부의 등분산성 검정 -> 
    # leven_stat :  0.011723736898379184 
    # leven_p :  0.915044305043978          유의수준 0.05 초과이므로 등분산성 성립

    two_sample = stats.ttest_ind(chong, young, equal_var = True)
    print('영업부, 총무부 독립검정 : ',two_sample)
    # 영업부, 총무부 독립검정 :  
    # TtestResult(statistic=np.float64(0.4585177708256519), 
    # pvalue=np.float64(0.6523879191675446), df=np.float64(17.0))
    # 독립검정 결과 p-value는 0.05 이상이므로 귀무가설을 채택한다.

    # 영업부와 총무부 직원들의 연봉 책정에는 부서와 관련이 없다.
except Exception as e:
    print('처리 오류 : ', e)
finally:
    conn.close()