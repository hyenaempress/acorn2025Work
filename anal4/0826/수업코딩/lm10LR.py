# sklearn 모듈의 LinearRegression 클래스 사용
import numpy as np           # 수치 계산 및 배열 연산을 위한 라이브러리
import pandas as pd          # 데이터프레임 기반 데이터 처리 라이브러리
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델 클래스
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error  # 회귀 평가 지표 함수들
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 데이터 표준화 및 정규화 클래스
import matplotlib.pyplot as plt

sample_size = 100            # 샘플 데이터 개수 설정
np.random.seed(1)            # 랜덤 시드 고정 (재현성 확보)

# 1) 편차가 있는 데이터 생성
x = np.random.normal(0, 10, sample_size)                  # 평균 0, 표준편차 10인 정규분포에서 x 데이터 생성
y = np.random.normal(0, 10, sample_size) + x * 30         # y는 x에 30을 곱해 강한 선형관계 추가 후 노이즈 더함
print(x[:5])                                              # x 데이터 앞부분 출력
print(y[:5])                                              # y 데이터 앞부분 출력
print('상관계수 : ', np.corrcoef(x, y))                   # x와 y의 상관계수(선형관계 정도) 출력

scaler = MinMaxScaler()                        
# MinMaxScaler 객체 생성 (0~1 정규화용)
x_scaled = scaler.fit_transform(x.reshape(-1, 1))         # x 데이터를 0~1 범위로 정규화
print('x_scaled:', x_scaled[:5])                          # 정규화된 x 데이터 앞부분 출력
# plt.scatter(x_scaled, y)                                  # 정규화된 x와 y의 산점도 시각화
# plt.show()                                                # 그래프 화면에 표시
# plt.close()
model = LinearRegression().fit(x_scaled, y)
print(model) # LinearRegression()

print('계수(slope) : ', model.coef_) # 회귀계수(각 독립 변수가 종속 변수에 미치는 영향력)
print('절편(intercept) : ', model.intercept_) 
print('결정 계수(R^2) : ', model.score(x_scaled, y)) # 설명력 : 훈련 데이터 기준
# y = wx + b <== 1350.4161554* x + -691.1877661754081

y_pred = model.predict(x_scaled) # 모델을 사용해 x_scaled에 대한 예측값 생성
print('예측값(y^)', y_pred[:5]) # [ 490.32381062 -182.64057041 -157.48540955 -321.44435455  261.91825779]
print('실제값(y)', y[:5]) # [ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]

# 선형회귀는 MAE(평균 절대 오차), MSE(평균 제곱 오차), RMSE(평균 제곱근 오차), R^2(결정 계수)로 평가
# 모델 성능 평가 함수 작성
def RegScoreFunc(y_true, y_pred):
    print('R^2_score(결정계수) : {}'.format(r2_score(y_true, y_pred))) # 절대적으로 사용
    print('설명분산점수: {}'.format(explained_variance_score(y_true, y_pred)))
    print('MSE(평균제곱오차) : {}'.format(mean_squared_error(y_true, y_pred)))

RegScoreFunc(y, y_pred)
# R^2_score(결정계수) : 0.9987875127274646
# 설명분산점수: 0.9987875127274646
# MSE(평균제곱오차) : 86.14795101998747

# ---

# 2) 편차가 꽤 있는 데이터 생성
x = np.random.normal(0, 1, sample_size)                  # 평균 0, 표준편차 10인 정규분포에서 x 데이터 생성(독립)
y = np.random.normal(0, 500, sample_size) + x * 30         # y는 x에 30을 곱해 강한 선형관계 추가 후 노이즈 더함(종속)
print(x[:5])                                              # x 데이터 앞부분 출력
print(y[:5])                                              # y 데이터 앞부분 출력
print('상관계수 : ', np.corrcoef(x, y))                   # x와 y의 상관계수(선형관계 정도) 출력

scaler = MinMaxScaler()                        
# MinMaxScaler 객체 생성 (0~1 정규화용)
x_scaled = scaler.fit_transform(x.reshape(-1, 1))         # x 데이터를 0~1 범위로 정규화
print('x_scaled:', x_scaled[:5])                          # 정규화된 x 데이터 앞부분 출력

model = LinearRegression().fit(x_scaled, y)
y_pred = model.predict(x_scaled) # 모델을 사용해 x_scaled에 대한 예측값 생성
print('예측값(y^)', y_pred[:5]) # [-10.75792685  -8.15919008 -11.10041394  -5.7599096  -12.73331002]
print('실제값(y)', y[:5]) # [1020.86531436 -710.85829436 -431.95511059 -381.64245767 -179.50741077]

RegScoreFunc(y, y_pred)
# R^2_score(결정계수) : 1.6093526521765433e-05
# 설명분산점수: 1.6093526521765433e-05
# MSE(평균제곱오차) : 282457.9703485092
# 결정계수가 거의 0에 가까워 설명력이 거의 없음. 즉, 선형회귀 모델이 데이터의 변동성을 거의 설명하지 못함.
# -> 편차가 큰 데이터는 선형회귀 모델이 잘 작동하지 않음.

