# 회귀분석 문제 5) 
# Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
# testdata 폴더 : Consumo_cerveja.csv
# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
# feature : Temperatura Media (C) : 평균 기온(C)
#             Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
# 조건 : NaN이 있는 경우 삭제!

# ----------------- 풀이 --------------------

# 1. 요구사항 분석
# - 종속 변수: 맥주 소비량 Consumo de cerveja (litros)
# - 독립 변수: 평균 기온 Temperatura Media (C), 강수량 Precipitacao
# - 회귀분석 모델 LinearRegression 작성
# - 결측치는 제거

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error  # 회귀 평가 지표 함수들
from sklearn.preprocessing import MinMaxScaler  # 데이터 표준화 및 정규화 클래스

# 2. 데이터 로딩
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Consumo_cerveja.csv")
# 데이터 확인용
print(data)
print(data.dtypes)

# 3. 필요한 데이터만 추출 & 결측치 제거
data = data[['Consumo de cerveja (litros)', 'Temperatura Media (C)', 'Precipitacao (mm)']].dropna()

# 4. 반점(,) -> 온점(.) 변환
for col in data.columns: # 컬럼을 하나씩 순회
    if data[col].dtype == 'object': # 타입이 object인 경우
        data[col] = data[col].str.replace(',', '.').astype(float) # 문자열에서 ','를 '.'로 변경 후 float형으로 변환

# 데이터 확인용
print(data)
print(data.dtypes)

# 5. 데이터 분할 : train, test - sort하면 안돼(왜곡된 자료로 분리)
train, test = train_test_split(data, test_size=0.4, random_state=1)
print(len(train), len(test))

x_train = train[['Temperatura Media (C)', 'Precipitacao (mm)']] # (독립 변수) 2차원 데이터 형태
y_train = train['Consumo de cerveja (litros)'] # (종속 변수) 1차원 데이터 형태
x_test = test[['Temperatura Media (C)', 'Precipitacao (mm)']]
y_test = test['Consumo de cerveja (litros)']

print(x_train)
print(y_train)
print(x_test)
print(y_test)

# 5. 회귀분석 모델 학습(x_train, y_train)
model = LinearRegression().fit(x_train, y_train) # (독립, 종속)

# 데이터 확인용
print("기울기 : ", model.coef_) # 0.80190566 -0.07366404 
print("절편 : ", model.intercept_) # 8.76264285047758

# 6. 모델 예측(입력: x_test, 실제 값: y_test)
prediction_result = model.predict(x_test) # 모델 평가(예측)는 test data를 사용
print("예측 결과 : ", np.round(prediction_result[:5])) # 25. 19. 25. 21. 30.
print("실제 결과 : ", y_test[:5].values.round(0)) # 25. 21. 24. 19. 30.

# 7. 모델 평가
# - **R² (결정계수, Coefficient of Determination)** | 0.3909297901333657
print('R^2_score(결정계수) : {}'.format(r2_score(y_test, prediction_result))) # 절대적으로 사용
# 기준: 1에 가까울 수록 좋음
# 해석: 모델이 실제 데이터의 분산 중 39% 설명

# - **MAE (Mean Absolute Error, 평균 절대 오차)** | 2.888090288817247
print('MAE(평균절대오차) : {}'.format(np.mean(np.abs(y_test - prediction_result))))
# 기준: 작을 수록 좋음.

# - **MSE (Mean Squared Error, 평균 제곱 오차)** | 12.044391351280687
print('MSE(평균제곱오차) : {}'.format(mean_squared_error(y_test, prediction_result)))
# 기준: 작을 수록 좋음.

# - **RMSE (Root Mean Squared Error, 평균 제곱근 오차)** | 3.470503040090973
print('RMSE(평균제곱근오차) : {}'.format(np.sqrt(mean_squared_error(y_test, prediction_result))))
# 기준: 작을 수록 좋음.

# 설명분산점수 | 0.4003082826487021
print('설명분산점수: {}'.format(explained_variance_score(y_test, prediction_result)))
# 기준: 1에 가까울 수록 좋음


# --------- 정규화 후 모델 생성 및 평가 -----------


# 1. 데이터 정규화
scaler = MinMaxScaler(feature_range=(-1, 1)) # 2차원 배열은 reshape로 정규화 불가
x_train_scaled = scaler.fit_transform(x_train.values)
x_test_scaled = scaler.transform(x_test.values)        # test 데이터는 transform만 사용

# 데이터 확인
print('x_train_scaled:\n', x_train_scaled[:5])
print('x_train: \n', x_train[:5])
print('x_test_scaled: \n', x_test_scaled[:5])
print('x_test: \n', x_test[:5])

# 2. 모델 학습
model = LinearRegression().fit(x_train_scaled, y_train) # (독립, 종속)
prediction_result = model.predict(x_test_scaled)
print("예측 결과 : ", np.round(prediction_result[:5])) # 예측 결과 :  [25. 19. 25. 21. 30.]
print("실제 결과 : ", y_test[:5].values.round(0)) # 실제 결과 :  [25. 21. 24. 19. 30.]

# 3. 모델 평가
print('R^2_score(결정계수) : {}'.format(r2_score(y_test, prediction_result))) # 0.3909297901333656

# 결론: 정규화를 거쳐도 차이가 없다.
# 정규화 전 결정계수: 0.3909297901333657
# 정규화 후 결정계수: 0.3909297901333656