# 분류 모델 성능 평가 관련

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
print(x[:3])
print(y)

# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

model = LogisticRegression().fit(x, y)
yhat = model.predict(x)
print(yhat)

f_value = model.decision_function(x) # 결정 함수(판별함수, 불확실성 추정함수), 판별 경계선 설정을 위한 샘플 자료 얻기
print("f_value: \n", f_value[:10])

df = pd.DataFrame(np.vstack([f_value, yhat, y]).T, columns=["f_value", "yhat", "y"])
print(df)

print(confusion_matrix(y, yhat))
acc = (44 + 44) / 100
recall = 44 / (44 + 4)
precision = 44 / (44 + 8)
specificity = 44 / (44 + 8)
fallout = 8 / (8 + 44) # 위양성률
print("acc(정확도): ", acc)
print("recall(재현율): ", recall)
print("precision(정밀도): ", precision)
print("specificity(특이도): ", specificity)
print("fallout(위양성률): ", fallout)
print("fallout(위양성률)", 1 - specificity)

ac_sco = metrics.accuracy_score(y, yhat)
print("ac_sco: ", ac_sco)
cl_rep = metrics.classification_report(y, yhat)
print("cl_rep: \n", cl_rep)
print()
fpr, tpr, thresholds = metrics.roc_curve(y,  model.decision_function(x)) # 임계값에 따른 fpr, tpr 값
print("fpr: ", fpr)
print("tpr: ", tpr)
print("분류임계결정값: ", thresholds)


# ROC 커브 시각화
plt.plot(fpr, tpr, 'o-', label = 'Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label = 'random classifier line(AUC 0.5)')
plt.plot([fallout], [recall], 'ro', ms = 10) # 위양성률과 재현율 값 출력
plt.xlabel('Fallout (False Positive Rate)')
plt.ylabel('Recall (True Positive Rate)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# AUC(Area Under the Curve) - ROC 커브의 면적
# 1에 가까울수록 좋은 분류모델로 평가됨
print('AUC : ', metrics.auc(fpr, tpr))