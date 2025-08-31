# --------------------------------------------------------------------------------
# [로지스틱 분류분석 문제3]
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 얘를 사용해도 됨   'testdata/advertisement.csv' 
# 참여 칼럼 : 
#    - Daily Time Spent on Site : 사이트 이용 시간 (분)
#    - Age : 나이,
#    - Area Income : 지역 소득,
#    - Daily Internet Usage :일별 인터넷 사용량(분),
#    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
# 모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
# 새로운 데이터로 분류 작업을 진행해 본다.
# --------------------------------------------------------------------------------

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로딩
df = pd.read_csv("./01_big_data_machine_learning/data/advertising.csv")

# 2. 데이터 확인
print(df.head())

# 3. 데이터 전처리
# 참여 칼럼 : 
#    - Daily Time Spent on Site : 사이트 이용 시간 (분)
#    - Age : 나이,
#    - Area Income : 지역 소득,
#    - Daily Internet Usage :일별 인터넷 사용량(분),
#    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )

# 필요없는 칼럼 제거
df = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Clicked on Ad']]

# 결측치 처리
df = df.dropna()

# 데이터 분할
X = df[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = df['Clicked on Ad']

# 4. 스케일링(StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 모델 학습
model = LogisticRegression()
model.fit(X_scaled, y)

# 6. 모델 예측
y_pred = model.predict(X_scaled)

# 7. 결정 함수
f_value = model.decision_function(X_scaled)
print(f_value)

# 8. 결과 데이터프레임 생성
df = pd.DataFrame(np.vstack([f_value, y_pred, y]).T, columns=["f_value", "yhat", "y"])

# 9. 혼동 행렬
conf_matrix = confusion_matrix(y, y_pred)
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
ac_sco = metrics.accuracy_score(y, y_pred)
print("ac_sco: ", ac_sco)

# 분류 리포트(Precision, Recall, F1 등) 출력
cl_rep = metrics.classification_report(y, y_pred)
print("cl_rep: \n", cl_rep)
print()

# ROC 곡선 계산: 임계값에 따른 FPR, TPR 값 반환
fpr, tpr, thresholds = metrics.roc_curve(y,  model.decision_function(X_scaled)) # 임계값에 따른 fpr, tpr 값
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