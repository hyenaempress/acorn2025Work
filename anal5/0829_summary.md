# 1. 혼동 행렬
|                | Predicted POSITIVE | Predicted NEGATIVE |
|:--------------:|:------------------:|:------------------:|
| Actual POSITIVE|        TP          |        FN          |
| Actual NEGATIVE|        FP          |        TN          |

# 2. 정밀도(Precision)
- **정밀도(Precision)**는 모델이 양성(POSITIVE)이라고 예측한 것 중에서 실제로 양성인 비율을 의미합니다.
- 계산식: **Precision = TP / (TP + FP)**
- TP: 실제 양성이고 예측도 양성 (True Positive)
- FP: 실제는 음성인데 예측을 양성으로 한 경우 (False Positive)

# 3. 특이도(Specificity)
- **특이도(Specificity)**는 실제 음성(NEGATIVE) 중에서 모델이 음성으로 정확하게 예측한 비율을 의미합니다.
- 계산식: **Specificity = TN / (TN + FP)**
- TN: 실제 음성이고 예측도 음성 (True Negative)
- FP: 실제는 음성인데 예측을 양성으로 한 경우 (False Positive)
- 특이도가 높을수록 모델이 음성을 잘 구분한다는 의미입니다.

# 4. 재현도(Recall) = 민감도(Sensitivity)
- **재현도(Recall)**는 실제 양성(POSITIVE) 중에서 모델이 양성으로 잘 맞춘 비율을 의미합니다.
- 계산식: **Recall = TP / (TP + FN)**
- TP: 실제 양성이고 예측도 양성 (True Positive)
- FN: 실제는 양성인데 예측을 음성으로 한 경우 (False Negative)

# 5. 정확도(Accuracy) (=정분류율)
- **정확도(Accuracy)**는 전체 데이터 중에서 모델이 맞게 예측한 비율을 의미합니다.
- 계산식: **Accuracy = (TP + TN) / (TP + TN + FP + FN)**
- TP: 실제 양성이고 예측도 양성 (True Positive)
- TN: 실제 음성이고 예측도 음성 (True Negative)
- FP: 실제는 음성인데 예측을 양성으로 한 경우 (False Positive)
- FN: 실제는 양성인데 예측을 음성으로 한 경우 (False Negative)
- 정확도가 높을수록 전체적으로 예측을 잘하는 모델입니다.

# 6. 성능 점수(F1 Score)
- **F1 Score**는 정밀도와 재현도의 조화 평균으로, 두 값의 균형을 평가합니다.
- 계산식: **F1 Score = 2 × (Precision × Recall) / (Precision + Recall)**
- F1 Score가 높을수록 정밀도와 재현도가 모두 좋은 모델입니다.

# 7. Scikit-Learn 제공 분류 성능 평가 지표(ROC curve) (https://cafe.daum.net/flowlife/SBU0/32)

