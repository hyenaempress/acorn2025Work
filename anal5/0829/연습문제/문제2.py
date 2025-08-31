# --------------------------------------------------------------------------------
# [로지스틱 분류분석 문제2] 
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값0(착용X), 값1(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X
# --------------------------------------------------------------------------------

# 입력: 게임, TV 시청 데이터
# 출력: 안경 착용 유무

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

# 1. 데이터 로딩
df = pd.read_csv("./01_big_data_machine_learning/data/bodycheck.csv")

# 2. 데이터 확인
print(df.head())

# 3. 데이터 전처리
df = df[['게임', 'TV시청', '안경유무']]
print(df)

# 4. 데이터 분류
x = df[['게임', 'TV시청']]
y = df['안경유무']

# 5. 모델 생성
model = LogisticRegression()
model.fit(x, y)

# 6. 모델 예측
yhat = model.predict(x)
print("실제값: \n", y.values)
print("예측값: \n", yhat)

# 7. 결정 함수
f_value = model.decision_function(x)
print(f_value)

# 8. 결과 데이터프레임 생성
df = pd.DataFrame(np.vstack([f_value, yhat, y]).T, columns=["f_value", "yhat", "y"])

# 9. 혼동 행렬
conf_matrix = confusion_matrix(y, yhat)
print("혼동 행렬: \n", conf_matrix)

# 10. 혼동 행렬 분석
print("TN: ", conf_matrix[0, 0]) # TN
print("FP: ", conf_matrix[0, 1]) # FP
print("FN: ", conf_matrix[1, 0]) # FN
print("TP: ", conf_matrix[1, 1]) # TP

acc = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[0, 1]) # 정확도 = True Positive / (True Positive + False Positive)
recall = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[1, 0]) # 민감도 = True Positive / (True Positive + False Negative)
precision = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[0, 1]) # 정밀도 = True Positive / (True Positive + False Positive)
specificity = (conf_matrix[0, 0]) / (conf_matrix[0, 0] + conf_matrix[0, 1]) # 특이도 = True Negative / (True Negative + False Positive)
fallout = (conf_matrix[0, 1]) / (conf_matrix[0, 1] + conf_matrix[0, 0]) # 위양성률 = False Positive / (False Positive + True Negative)
print("acc(정확도): ", acc)
print("recall(재현율): ", recall)
print("precision(정밀도): ", precision)
print("specificity(특이도): ", specificity)
print("fallout(위양성률): ", fallout) # 음성 중에서 잘못 양성으로 예측한 비율
print("fallout(위양성률): ", 1 - specificity)

# 11. 추가 지표
# 정확도(Accuracy) 점수 출력
ac_sco = metrics.accuracy_score(y, yhat)
print("ac_sco: ", ac_sco)

# 분류 리포트(Precision, Recall, F1 등) 출력
cl_rep = metrics.classification_report(y, yhat)
print("cl_rep: \n", cl_rep)
print()

# ROC 곡선 계산: 임계값에 따른 FPR, TPR 값 반환
fpr, tpr, thresholds = metrics.roc_curve(y,  model.decision_function(x)) # 임계값에 따른 fpr, tpr 값
print("fpr: ", fpr)         # False Positive Rate (위양성률)
print("tpr: ", tpr)         # True Positive Rate (재현율)
print("분류임계결정값: ", thresholds) # 각 점에서의 임계값

# 12. 시각화
plt.plot(fpr, tpr, 'o-', label = 'Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label = 'random classifier line(AUC 0.5)')
plt.plot([fallout], [recall], 'ro', ms = 10) # 위양성률과 재현율 값 출력
plt.xlabel('Fallout (False Positive Rate)')
plt.ylabel('Recall (True Positive Rate)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 13. AUC(Area Under the Curve) - ROC 커브의 면적
# 1에 가까울수록 좋은 분류모델로 평가됨
print('AUC : ', metrics.auc(fpr, tpr))