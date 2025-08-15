# https://cafe.daum.net/flowlife/SBU0/23

import pandas as pd
import scipy.stats as stats
# 시각화: 콘솔로만 찍기
# * 카이제곱 검정
# print('--------------문제1--------------')
# # 카이제곱 문제1) 부모학력 수준이 자녀의 진학여부와 관련이 있는가?를 가설검정하시오
#   예제파일 : cleanDescriptive.csv
#   칼럼 중 level - 부모의 학력수준, pass - 자녀의 대학 진학여부
#   조건 :  level, pass에 대해 NA가 있는 행은 제외한다.
 
# 귀무 가설: 부모학력 수준이 자녀의 진학여부와 관련이 없다
# 대립 가설: 부모학력 수준이 자녀의 진학여부와 관련이 있다
data=pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv')
#print(data)
# NA가 있는 행은 제외한다.
data = data.dropna(subset=['level', 'pass'])
ctab=pd.crosstab(index=data['level'],columns=data['pass'])
print(ctab)
print()
# p-value 
chi2,p,dof,_=stats.chi2_contingency(ctab)
print(f'카이제곱:{chi2}, 피밸류:{p}, 자유도:{dof}')
# 유의 수준: 0.05 카이제곱:2.7669512025956684, 피밸류:0.25070568406521365, 자유도:2
# 결과: p 0.2507 >  알파 0.05 -> 귀무가설 채택.  부모학력 수준이 자녀의 진학여부와 관련이 없다
 
 
print('--------------문제2--------------') 
# 카이제곱 문제2) 지금껏 A회사의 직급과 연봉은 관련이 없다. 
# 그렇다면 jikwon_jik과 jikwon_pay 간의 관련성 여부를 통계적으로 가설검정하시오.
#   예제파일 : MariaDB의 jikwon table 
#   jikwon_jik   (이사:1, 부장:2, 과장:3, 대리:4, 사원:5)
#   jikwon_pay (1000 ~2999 :1, 3000 ~4999 :2, 5000 ~6999 :3, 7000 ~ :4)
#   조건 : NA가 있는 행은 제외한다.  
# 귀무 가설: 지금껏 A회사의 직급과 연봉은 관련이 없다.
# 대립 가설: 지금껏 A회사의 직급과 연봉은 관련이 있다. 

import MySQLdb
import pickle
import sys
import numpy as np
from scipy import stats        # ✅ chi2_contingency 사용

# DB 연결 함수
def get_conn(cfg):
    return MySQLdb.connect(**cfg)

# 1) DB 접속
try:
    with open(r'C:\Users\SeYun\anaconda3\envs\day0731\mymaria.dat', 'rb') as obj:
        config = pickle.load(obj)
except Exception as e:
    print('!!!!!!!!!! 트라이1 처리오류 !!!!!!!!!!:', e)
    sys.exit(1)

# 2) 데이터 읽기 → 전처리 → 교차표 → 카이제곱 검정
try:
    with get_conn(config) as conn:
        sql = """
        SELECT jikwonjik AS '직급', jikwonpay AS '연봉'
        FROM jikwon
        ;
        """
        df = pd.read_sql(sql, conn)

        # ---- 전처리 ----
        # 결측 제거 (직급/연봉 둘 중 하나라도 NaN이면 제거)
        df = df.dropna(subset=['직급', '연봉'])

        # 직급을 숫자 등급으로 매핑 (순서 보장 목적)
        jik_map = {'이사': 1, '부장': 2, '과장': 3, '대리': 4, '사원': 5} 
        df['직급'] = df['직급'].replace(jik_map)
        # 연봉 구간화
        # 구간: (1000~2999]=1, (2999~4999]=2, (4999~6999]=3, (6999~∞)=4
        # right=True → 오른쪽 경계 포함
        bins   = [1000, 2999, 4999, 6999, np.inf] # np.inf ∞
        labels = [1, 2, 3, 4]
        df['연봉'] = pd.cut(
            df['연봉'],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True # include_lowest=True → 최소값 포함
        )

        # ---- 교차표 작성 ----
        # 원본 교차표(존재하는 수준만 계산)
        base_ctab = pd.crosstab(df['직급'], df['연봉'])

        # 보기 좋게 5x4 고정(없는 수준은 0으로 채움)
        ctab = base_ctab.reindex(
            index=[1, 2, 3, 4, 5],
            columns=[1, 2, 3, 4],
            fill_value=0
        )
        print('교차표(관측도수):\n', ctab, '\n')
        print()
        # ---- 카이제곱 검정 ----
        # 주의: 모든 값이 0인 행/열은 기대도수가 0이 되어 부적절하므로 검정에서 제외
        row_mask = ctab.sum(axis=1) > 0
        col_mask = ctab.sum(axis=0) > 0
        test_tab = ctab.loc[row_mask, col_mask]
        chi2, p, dof, expected = stats.chi2_contingency(test_tab)

        # 기대도수 표를 원래 정렬로 맞춰서 보기 좋게 출력(검정은 test_tab 기준으로 계산됨)
        expected_df = pd.DataFrame(expected, index=test_tab.index, columns=test_tab.columns)
        print(f'카이제곱: {chi2:.3f}, 피밸류: {p:.3f}, 자유도: {dof}')
        print('기대도수표:\n', expected_df)
        print()
        
        # 해석(유의수준 0.05 기준)
        if 0.05 >=p :
            print('!!!!!!!!!!!!!!',type(p))
            print('결론: 귀무가설 기각, 대립 가설 채택')
        else:
            print('결론: 귀무가설 채택')

except Exception as e:
    print('!!!!!!!!!! 트라이2 처리오류 !!!!!!!!!!:', e)
    conn.close()
    # with 블록이라 conn은 자동 종료됨