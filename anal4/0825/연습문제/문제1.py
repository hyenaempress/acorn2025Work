# 회귀분석 문제 2)
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다.
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측

import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기
students = pd.read_csv("./01_big_data_machine_learning/data/student.csv")

# 2. 국어 점수만으로 수학 점수 예측하는 단순 선형회귀 모델 생성
result1 = smf.ols(formula='수학 ~ 국어', data=students).fit()
# print(result1.summary())  # 모델 요약 출력(해석 참고)

# # 국어 점수 입력 받아 수학 점수 예측 (실행 시 주석 해제)
# input1 = float(input("국어 점수를 입력하시면, 수학 점수를 예측해 드립니다. \n 국어 점수 입력: "))
# print("예측된 수학 점수: ", result1.predict(pd.DataFrame({'국어': [input1]})).values)

# 3. 국어, 영어 점수로 수학 점수 예측하는 다중 선형회귀 모델 생성
result2 = smf.ols(formula='수학 ~ 국어 + 영어', data=students).fit()
print(result2.summary())  # 모델 요약 출력

# # 국어, 영어 점수 입력 받아 수학 점수 예측 (실행 시 주석 해제)
# input2 = float(input("국어 점수 입력: "))
# input3 = float(input("영어 점수 입력: "))
# print("예측된 수학 점수: ", result2.predict(pd.DataFrame({'국어': [input2], '영어': [input3]})).values)

# 4. 1번째 모델(국어만) 검정 결과 출력
print("1번째 모델 검정 결과: ")
print("R-squared: ", result1.rsquared)  # 설명력(국어 점수로 수학 점수의 약 58.7% 설명)
print("p-value: ", result1.pvalues.iloc[1])  # 국어 점수의 p-value(유의성 검정)
# p-value < 0.05 이므로 국어 점수와 수학 점수 사이에 유의한 선형 관계가 있음

# 5. 2번째 모델(국어+영어) 검정 결과 출력
print("\n2번째 모델 검정 결과: ")
print("R-squared: ", result2.rsquared)  # 설명력(국어, 영어 점수로 수학 점수의 약 65.9% 설명)
print("Prob (F-statistic):", result2.f_pvalue)  # 모델 전체의 유의성 검정 p-value
# Prob (F-statistic) < 0.05 이므로 국어, 영어 점수와 수학 점수 사이에 유의한 선형 관계가 있음

# 결론:
# - 국어 점수만으로도 수학 점수 예측에 중간~높은 수준의 설명력을 보임
# - 국어와 영어 점수를 함께 사용하면 예측력이 더 좋아짐
# - 단, 각 변수의 p-value도 함께 확인하여 실제로 유의미한 변수만 사용하는 것이 바람직함
