# https://cafe.daum.net/flowlife/SBU0/44

# django 사용 X 
# 배운방법 다 응용해서 이해하며 천천히 풀어라 

# [로지스틱 분류분석 문제1]
# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라. 
# [정리]
# - 독립변수: 소득수준 (연속형, 키보드 입력)
# - 종속변수: 외식유무 (범주형, 소득 수준 → 외식 여부(1=외식, 0=비외식) 분류)

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # 모델이 단순히 외운 건지(오버피팅), 아니면 새로운 데이터에도 잘 맞는지(일반화) 확인

# 1) 데이터 불러오기
data = pd.read_csv('./01_big_data_machine_learning/code/0828/수업코딩/logi2_Quiz.csv') 
print(data.head())

# model 학습은 train 데이터 , 평가(혼란표/정확도)는 test 데이터 
#  test 3 train 7 분리 (3:7 비율)
train, test = train_test_split(data, test_size=0.3, random_state=42)

# -------------------------------------------------
# 방법1) logit() : 로지스틱 회귀 전용
# -------------------------------------------------
formula = '외식유무 ~ 소득수준'
logit_model = smf.logit(formula=formula, data=train).fit() # 학습
#print("\n[Logit 방식 요약]:\n", logit_model.summary())

# 혼란표 (pred_table은 학습데이터 기준)
conf_tab1 = logit_model.pred_table()
print("\n[Logit 혼란표 conf_tab1]:\n", conf_tab1)
# 분류 정확도 (맞춘 개수 / 전체 개수)
print("\n Logit 분류정확도 conf_tab1 :\n", (conf_tab1[0][0] + conf_tab1[1][1]) / len(train))

pred1 = logit_model.predict(test) # 예측
print("\n Logit 분류정확도 accuracy_score:\n", accuracy_score(test['외식유무'], np.around(pred1)))

# -------------------------------------------------
# 방법2) glm() : 일반화된 선형모델 (Binomial → 로지스틱)
# -------------------------------------------------
glm_model = smf.glm(formula=formula, data=train, family=sm.families.Binomial()).fit()
#print("\n[GLM 방식 요약]:\n", glm_model.summary())

# glm은 pred_table() 지원 안됨
pred2 = glm_model.predict(test)
pred_class2 = np.around(pred2)

print("\nGLM 분류정확도:\n", accuracy_score(test['외식유무'], pred_class2))

# -------------------------------------------------
# 참고) 혼란표 confusion_matrix
# -------------------------------------------------
from sklearn.metrics import confusion_matrix
print("\n[ Logit 혼란표 confusion_matrix ]:\n", confusion_matrix(test['외식유무'], np.around(pred1)))
print("\n[GLM 혼란표]:\n", confusion_matrix(test['외식유무'], pred_class2))

# -------------------------------------------------
# 새로운 소득 입력 받아 분류 예측
# -------------------------------------------------

x = int(input("\n소득 수준을 입력하세요 (정수): "))
new_df = pd.DataFrame({'소득수준':[x]})

prob1 = logit_model.predict(new_df)[0]
prob2 = glm_model.predict(new_df)[0]

print(f"\n입력한 소득: {x}")
print(f" [Logit] 외식확률={prob1:.4f}, 분류결과={np.rint(prob1)}")
print(f" [GLM]   외식확률={prob2:.4f}, 분류결과={np.rint(prob2)}")


