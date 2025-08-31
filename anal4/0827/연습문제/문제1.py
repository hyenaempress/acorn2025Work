# 다항회귀분석 문제) 

# 데이터 로드 (Servo, UCI) : "https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data"
# cols = ["motor", "screw", "pgain", "vgain", "class"]

#  - 타깃/피처 (숫자만 사용: pgain, vgain)
#    x = df[["pgain", "vgain"]].astype(float)   
#    y = df["class"].values
#  - 학습/테스트 분할 ( 8:2 )
#  - 스케일링 (StandardScaler)
#  - 다항 특성 (degree=2) + LinearRegression 또는 Ridge 학습
#  - 성능 평가 
#  - 시각화

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # 다항회귀 모델 작성용
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
from sklearn.decomposition import PCA # 차원 축소용
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
# 1. 데이터 준비
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data", header = None)
print(df)

df.columns = ['motor', 'screw', 'pgain', 'vgain', 'class']
print(df)

# 2. 필요 데이터 추출
x = df[["pgain", "vgain"]].astype(float) # 타깃/피처 (숫자만 사용: pgain, vgain)
y = df["class"].values

# 3. 학습/테스트 분할( 8:2 )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. 스케일링(StandardScaler)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 5. 다항회귀 모델 작성 : 다향 특성(degree = 2)
model = LinearRegression().fit(x_train_scaled, y_train)
quad = PolynomialFeatures(degree=2)
x_quad = quad.fit_transform(x_train_scaled)
model.fit(x_quad, y_train)

# 6. 다항회귀 모델 예측
y_pred = model.predict(quad.transform(x_test_scaled))
print("예측 값: \n", y_pred)
print("실제 값: \n", y_test)

# 7. 성능 평가
q_r2 = r2_score(y_test, y_pred)
print('다항(2)회귀 결정 계수(R²): ', q_r2) # 0.54 (54%)
print("MSE:\t", mean_squared_error(y_test, y_pred)) # 0.97
print("설명력:\t", r2_score(y_test, y_pred)) # 0.54

# 8. 시각화: x의 인자가 pgain과 vgain 2개로, 원하는 형태의 그래프가 나오지 않아 3D 산점도로 변경

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# meshgrid 생성
x_range = np.linspace(x_test_scaled[:, 0].min(), x_test_scaled[:, 0].max(), 30)
y_range = np.linspace(x_test_scaled[:, 1].min(), x_test_scaled[:, 1].max(), 30)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# meshgrid 좌표를 (N, 2)로 변환
xy_mesh = np.column_stack([x_mesh.ravel(), y_mesh.ravel()])

# 다항 특성 변환 후 예측값 계산
xy_mesh_poly = quad.transform(xy_mesh)
z_mesh = model.predict(xy_mesh_poly).reshape(x_mesh.shape)

ax.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], y_test, c='b', marker='o', alpha=0.6)
ax.plot_surface(x_mesh, y_mesh, z_mesh, color='r', alpha=0.5)
ax.set_title('3D 산점도: pgain, vgain, class')

plt.show()
plt.close()