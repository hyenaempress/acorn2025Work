# 선형회귀 평가 지표 관련
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error  # 회귀 평가 지표 함수들

# 공부 시간에 따른 시험 점수 데이터 : 표본 수 16개
df = pd.DataFrame({
    'studytime' : [3, 4, 5, 8, 10, 5, 8, 6, 3, 6, 10, 9, 7, 0, 1, 2], 
    'score' : [76, 74, 74, 89, 66, 75, 84, 82, 73, 81, 95, 88, 83, 40, 70, 69] 
})
print(df.head(3))

# dataset 분리 : train, test - sort하면 안돼(왜곡된 자료로 분리)
train, test = train_test_split(df, test_size=0.4, random_state=1)
print(len(train), len(test))

x_train = train[['studytime']] # (독립 변수) 2차원 데이터 형태
y_train = train['score'] # (종속 변수) 1차원 데이터 형태
x_test = test[['studytime']]
y_test = test['score']

print(x_train)
print(y_train)
print(x_test)
print(y_test)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (8, 1) (8, 1) (8,) (8,)
print()

model = LinearRegression()
model.fit(x_train, y_train) # 모델 학습은 train data를 사용
y_pred = model.predict(x_test) # 모델 평가(예측)는 test data를 사용

print('예측값 : ',  np.round(y_pred)) # 예측값 :  [85. 66. 80. 78. 85. 90. 90.]
print('실제값 : ', y_test.values) # 실제값 :  [89 40 82 74 84 95 66]

# 이제 위 예측 값이 정확한 지 확인하기 위해 아래와 같은 평가 지표를 사용하여 모델을 평가 함.

# - **MAE (Mean Absolute Error, 평균 절대 오차)**
#   - 실제값과 예측값의 차이의 절대값 평균
#   - 직관적으로 오차의 크기를 해석할 수 있음
#   - 이상치에 덜 민감함

# - **MSE (Mean Squared Error, 평균 제곱 오차)**
#   - 실제값과 예측값의 차이를 제곱하여 평균
#   - 큰 오차에 더 큰 패널티를 부여 (이상치에 민감)
#   - 모델 최적화 시 주로 사용되는 손실 함수

# - **RMSE (Root Mean Squared Error, 평균 제곱근 오차)**
#   - MSE에 제곱근을 취한 값
#   - 실제 데이터 단위와 동일해 해석이 쉬움
#   - 오차가 클수록 더 크게 반영

# - **R² (결정계수, Coefficient of Determination)**
#   - 모델이 실제 데이터를 얼마나 잘 설명하는지 나타내는 지표 (0~1 사이)
#   - 1에 가까울수록 예측력이 높음
#   - 전체 분산 중 모델이 설명하는 분산의 비율

print('모델의 성능은?')
# 결정 계수 공식
# R² = 1 - (Σ(yᵢ - ŷᵢ)²) / (Σ(yᵢ - ȳ)²)
#   - yᵢ : 실제값
#   - ŷᵢ : 예측값
#   - ȳ : 실제값의 평균
# R² = 1 - (Σ(실제값 - 예측값)²) / (Σ(실제값 - 평균값)²)
def R_squared(real, estimate):
   sum_of_squared_errors = np.sum(np.square(real - estimate)) # 오차 제곱 합(SSE)
   sum_of_squares_total = np.sum(np.square(real - np.mean(real))) # 편차 제곱 합(SST)
   return 1 - (sum_of_squared_errors / sum_of_squares_total) if sum_of_squares_total != 0 else 0

# 결정계수 수식으로 직접 작성 후 api 메소드와 비교
# 잔차 구하기
y_mean = np.mean(y_test) # y의 평균
# 오차 제곱 합(실제 값과 예측된 결과의 오차를 제곱한 총합) : Σ(실제값 - 예측값)²
bunja = np.sum(np.square(y_test - y_pred))
# 편차 제곱 합(실제 값과 평균 값 과의 차이를 제곱한 총합) : Σ(실제값 - 평균값)²
bunmo = np.sum(np.square(y_test - y_mean))
r2 = 1 - bunja / bunmo # 1 - (오차 제곱 합 / 편차 제곱 합)
print("계산된 결정 계수: ", r2)

# - **R² (결정계수, Coefficient of Determination)**
print('R^2_score(결정계수) : {}'.format(r2_score(y_test, y_pred))) # 절대적으로 사용

# - **MAE (Mean Absolute Error, 평균 절대 오차)**
print('MAE(평균절대오차) : {}'.format(np.mean(np.abs(y_test - y_pred))))

# - **MSE (Mean Squared Error, 평균 제곱 오차)**
print('MSE(평균제곱오차) : {}'.format(mean_squared_error(y_test, y_pred)))

# - **RMSE (Root Mean Squared Error, 평균 제곱근 오차)**
print('RMSE(평균제곱근오차) : {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

# 설명분산점수
print('설명분산점수: {}'.format(explained_variance_score(y_test, y_pred)))

# R^2 값은 분산을 기반으로 측정하는 지표로, 중심 극한 정리에 의해 표본 데이터가 많아져 정규 분포를 따를 때 그 수치도 증가한다.
import seaborn as sns
import matplotlib.pyplot as plt

def linearFunc(df, test_size):
    train, test = train_test_split(df, train_size = test_size, shuffle = True, random_state = 2)
    x_train = train[['studytime']]
    y_train = train['score']
    x_test = test[['studytime']]
    y_test = test['score']

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # R^2 계산
    print('R제곱값 : ', round(r2_score(y_test, y_pred), 2))
    print("너 누구니? : ", i)
    print('test data 비율 : 전체 데이터 수의 {0}%'.format(i * 100))
    print('데이터 수 : {0}개 '.format(x_train))

    # 시각화
    # sns.scatterplot(x=df['studytime'], y=df['score'], color='green')
    # sns.scatterplot(x=x_test['studytime'], y=y_test, color='red')
    # sns.lineplot(x=x_test['studytime'], y=y_pred, color='blue')
    # plt.show()
    # plt.close()

# test 자료 수를 10%에서 50%로 늘려가며 R^2 값 구하기
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5] 
for i in test_sizes:
   linearFunc(df, i) # 점차 test 자료 수가 늘어나는 값을 입력으로 준다.
