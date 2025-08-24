# kind quantity
# 1 64
# 2 72
# 3 68
# 4 77
# 2 56
# 1 NaN
# 3 95
# 4 78
# 2 55
# 1 91
# 2 63
# 3 49
# 4 70
# 1 80
# 2 90
# 1 33
# 1 44
# 3 55
# 4 66
# 2 77

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm    # 방법 2의 아노바 쓰려고
from statsmodels.formula.api import ols         # 방법 2의 아노바 쓰려고
from statsmodels.stats.multicomp import pairwise_tukeyhsd   # 이건 사후검정
import pickle
import sys
import MySQLdb

# [ANOVA 예제 1]
# 빵을 기름에 튀길 때 네 가지 기름의 종류에 따라 빵에 흡수된 기름의 양을 측정하였다.
# 기름의 종류에 따라 흡수하는 기름의 평균에 차이가 존재하는지를 분산분석을 통해 알아보자.
# 조건 : NaN이 들어 있는 행은 해당 칼럼의 평균값으로 대체하여 사용한다.

data = {
    'kind' : [1,2,3,4,2,1,3,4,2,1,2,3,4,1,2,1,1,3,4,2],
    'quantity' : [64,72,68,77,56,np.nan,95,78,55,91,63,49,70,80,90,33,44,55,66,77]
}

# 귀무가설(H0) : 빵을 기름에 튀길 때, 서로 다른 네 종류의 기름이 빵에 흡수되는 양은 빵 종류에 관련이 없다.
# 대립가설(H1) : 기름의 종류에 따라 빵에 흡수되는 기름의 양에 차이가 있다.

# 결측치에 넣을 평균값 구하기
# 사용한 quan_mean 변수는 결측치에 채우기만 하면 다신 안쓸거야
df = pd.DataFrame(data)
print('결측치 채우기 전의 df : \n',df)
quan_mean = df.dropna()
quan_mean = df['quantity'].mean()
print('결측치에 채워줄 퀀티티 평균값 : \n',quan_mean)

df = df.fillna(quan_mean)
print('결측치를 평균값으로 채운 후의 df : \n',df)


# 방법 1 : f_oneway 함수 써서 햇어 그래서 멀쩡한 kind 칼럼을 그룹 네개로 나눠서 햇어
# f_oneway 쓰려면 그룹이 두 개 이상이 필요하대
group1 = df[df['kind'] == 1]['quantity']
group2 = df[df['kind'] == 2]['quantity']
group3 = df[df['kind'] == 3]['quantity']
group4 = df[df['kind'] == 4]['quantity']
f_result = stats.f_oneway(group1, group2, group3, group4)

print('stats.f_oneway : \n', f_result)
print('-' * 100)
#  F_onewayResult
# (statistic=np.float64(0.26693511759829797), 
# pvalue=np.float64(0.8482436666841788))    p-value > 유의수준(0.05) 따라서 귀무채택이야

# 방법 2 : anova 써서 해보자
# lmodel = ols('quantity ~ C(kind)', data = df).fit()   
# print(anova_lm(lmodel, type = 2))             # 이렇게 바로 뽑아보지 말고 변수 하나 지정해서 
print('-' * 100)                                # 1차원 배열로 넘겨주래

lmodel = ols('quantity ~ C(kind)', data = df).fit()
anova_result = anova_lm(lmodel, type = 2)
print('anova_result : \n', anova_result) 

#              df       sum_sq     mean_sq         F    PR(>F)  <-------- 이게 피밸류래
# C(kind)    3.0   231.304247   77.101416  0.266935  0.848244
# Residual  16.0  4621.432595  288.839537       NaN       NaN

# p-value = 0.848244 > 0.05 따라서 귀무채택 맞아
# 그럼 이제 사후검정 해보자

turkyResult = pairwise_tukeyhsd(endog = df.quantity, groups = df.kind)
print(turkyResult)
print('-' * 100)
#  Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# =====================================================
# group1 group2 meandiff p-adj   lower    upper  reject
# -----------------------------------------------------
#      1      2   5.5789   0.94  -22.494 33.6519  False
#      1      3   3.4956 0.9884 -27.8909 34.8822  False
#      1      4   9.4956 0.8223 -21.8909 40.8822  False
#      2      3  -2.0833 0.9975 -33.4699 29.3032  False
#      2      4   3.9167 0.9838 -27.4699 35.3032  False
#      3      4      6.0 0.9581 -28.3822 40.3822  False
# reject에 다 False 나오는거 보니까 위의 귀무채택 결과와 상응해
# H0 : 빵에 기름이 흡수되는 정도는 기름의 종류에 상관관계가 없다.

# [ANOVA 예제 2]

# DB에 저장된 buser와 jikwon 테이블을 이용하여 
# 총무부, 영업부, 전산부, 관리부 직원의 연봉의 평균에 차이가 있는지 검정하시오. 
# 만약에 연봉이 없는 직원이 있다면 작업에서 제외한다.

# 귀무가설(H0) : 전 부서 직원들의 연봉은 부서에 따라 다른 것이 아니다.
# 대립가설(H1) : 전 부서 직원들의 연봉은 부서에 따라 매겨지는 것이다.

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('로드 오류 : ', e)

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()

    sql = """
        select jikwonpay as 연봉, busername as 부서명
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
        """
    cursor.execute(sql)

    df = pd.DataFrame(cursor.fetchall(),
                        columns = ['연봉', '부서명']
                        )
    print(data)
    print('-' * 100)

    # 여기까지 디비 가져왓고 이제 이거갖고 1번처럼 두 방법으로 검정하고 
    # 사후검정까지 해보자
    # 아 그리고 당연히 월급 못받는 직원은 없다

    group_a = df[df['부서명'] == '총무부']['연봉']
    group_b = df[df['부서명'] == '영업부']['연봉']
    group_c = df[df['부서명'] == '전산부']['연봉']
    group_d = df[df['부서명'] == '관리부']['연봉']

    







except Exception as e:
    print('처리 오류 : ',e)

finally:
    conn.close()