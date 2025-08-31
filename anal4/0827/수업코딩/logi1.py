# Logistic Regression
# 독립(feature, x): 연속형
# 종속(label, y): 범주형
# 이항 분류(다항도 가능)
# 출력된 연속형(확률) 자료를 logit 변환하여, 최종적으로 sigmoid function에 의해 0 ~ 1 사이의 실수 값이
# 나오는데 0.5를 기준으로 0과 1로 분류합니다.
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf


# sigmoid function 살짝 맛보기
def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

# print(sigmoidFunc(3))
# print(sigmoidFunc(1))
# print(sigmoidFunc(-123))
# print(sigmoidFunc(0.123))

# mtcar dataset 사용
mtcardata = sm.datasets.get_rdataset('mtcars')
print(mtcardata.keys())
mtcars=mtcardata.data
print(mtcars.head(2))  
#                 mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
# rownames
# Mazda RX4      21.0    6  160.0  110   3.9  2.620  16.46   0   1     4     4
# Mazda RX4 Wag  21.0    6  160.0  110   3.9  2.875  17.02   0   1     4     4 
# am 은 오토 매뉴얼 (자동수동)
# mpg, hp가 am에 영향을 준다
mtcar=mtcars.loc[:,['mpg','hp','am']]
print(mtcar.head(2))  
#                 mpg   hp  am
# rownames
# Mazda RX4      21.0  110   1
# Mazda RX4 Wag  21.0  110   1 
print(mtcar['am'].unique()) # [1 0]

# 연비와 마력수에 따른 변속기 분류 모델 작성(수동, 자동)
# 모델 작성 방법1: logit()
formula = 'am ~ hp + mpg'
model1 = smf.logit(formula=formula, data=mtcar).fit()
print(model1.summary()) # Logit Regression Results

#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:                     am   No. Observations:                   32
# Model:                          Logit   Df Residuals:                       29
# Method:                           MLE   Df Model:                            2
# Date:                Wed, 27 Aug 2025   Pseudo R-squ.:                  0.5551
# Time:                        17:07:56   Log-Likelihood:                -9.6163
# converged:                       True   LL-Null:                       -21.615
# Covariance Type:            nonrobust   LLR p-value:                 6.153e-06
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
# hp             0.0550      0.027      2.045      0.041       0.002       0.108
# mpg            1.2596      0.567      2.220      0.026       0.147       2.372

# 예측값 / 실제값 출력
import numpy as np
# print('예측값: ', model1.predict())
pred = model1.predict(mtcar[:10])
print('예측값: ', np.around(pred.values))
print('실제값: ', mtcar['am'][:10].values)
print()

# 분류 모델의 정확도(accuracy) 확인
conf_tab = model1.pred_table()
print("confusion matrix: \n", conf_tab)

# confusion matrix
# [16.] 맞은 걸 맞았다고 한 개수
# [ 3.] 맞은 걸 틀렸다고 한 개수
# [ 3.] 틀린 걸 맞았다고 한 개수
# [10.] 틀린 걸 틀렸다고 한 개수

print("분류 정확도: ", (16 + 10)/len(mtcar)) # 모델이 맞춘 개수 / 전체 개수
print("분류 정확도: ", (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))
