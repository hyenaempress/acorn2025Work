## 1. 다중공선성(https://ysyblog.tistory.com/171)
- 다중공선성
- 상관관계가 매우 높은 독립변수들이 동시에 모델에 포함될 때 발생
- 만약 두 변수가 완벽하게 다중공선성에 걸려있으면, 같은 변수를 두번넣은 것이며 최소제곱법을 계산하는 것이 어렵다.
- 완벽한 다중공선성이 아니더라도 다중공선성이 높다면 회귀계수의 표준오차가 비정상적으로 커지게된다.
- 회귀계수의 유의성은 t-값에 의해 계산되는데(회귀계수 / 표준오차) 다중공선성으로 인해 표준오차가 비정상적으로 커지면 t값이 -
- 작아져서 p값이 유의해지지 않아 유의해야할 변수가 유의하지 않게됨.
- 회귀계수(베타)값을 제대로 측정하지 못하게 됨.

### 1.1 독립 변수와 종속 변수
- 독립 변수 끼리는 서로 상관 관계가 **없어야(낮아야)** 함
- 독립 변수는 종속 변수와 상관 관계가 **있어야(높아야)** 함

### 1.2 독립 변수끼리 상관 관계가 높으면?
- **다중공선성**이 발생하여 회귀계수의 추정이 불안정해질 수 있다.
- 따라서, **차원 축소(예: PCA)**를 하거나 **변수를 선택하는 방법(예: Lasso 회귀)**을 고려해야 한다.

#### 1) PCA란? (Principal Component Analysis, 주성분 분석)
- **정의**: 고차원 데이터를 저차원으로 축소하면서 데이터의 분산을 최대한 보존하는 차원 축소 기법
- **목적**: 
  - 다중공선성 문제 해결
  - 차원의 저주(curse of dimensionality) 완화
  - 시각화 및 계산 효율성 향상

##### PCA 작동 원리
```
1단계: 데이터 표준화 (평균 0, 분산 1로 조정)
2단계: 공분산 행렬 계산
   Cov(X) = (1/n-1) × X^T × X
3단계: 고유값(eigenvalue)과 고유벡터(eigenvector) 계산
4단계: 고유값 크기 순으로 정렬
5단계: 상위 k개 주성분 선택
6단계: 원본 데이터를 주성분 공간으로 변환
```

##### 주성분의 의미
- **제1주성분(PC1)**: 데이터 분산을 가장 많이 설명하는 방향
- **제2주성분(PC2)**: PC1과 직교하면서 두 번째로 분산을 많이 설명하는 방향
- **제k주성분(PCk)**: 이전 주성분들과 모두 직교하면서 k번째로 분산을 많이 설명하는 방향

##### 다중공선성 해결 메커니즘
- **직교성**: 주성분들은 서로 완전히 독립적 (상관계수 = 0)
- **분산 집중**: 원본 변수들의 정보를 소수의 주성분으로 압축
- **선형 결합**: 각 주성분은 원본 변수들의 선형 결합

##### Python 구현 예시
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용
pca = PCA(n_components=0.95)  # 95% 분산 보존
X_pca = pca.fit_transform(X_scaled)

# 주성분별 기여도 확인
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

print(f"선택된 주성분 개수: {pca.n_components_}")
print(f"설명된 분산 비율: {explained_variance_ratio}")
```

##### 장단점
**장점:**
- 다중공선성 완전 제거
- 차원 축소로 계산 효율성 향상
- 노이즈 감소 효과

**단점:**
- 해석 가능성 저하 (주성분의 의미가 모호)
- 원본 변수의 중요도 파악 어려움
- 선형 관계만 고려 가능

#### 2) Lasso 회귀 (Least Absolute Shrinkage and Selection Operator)
- **정의**: L1 정규화를 사용하여 회귀계수를 축소하고 변수 선택을 자동으로 수행하는 회귀 기법
- **목적**:
  - 다중공선성 문제 해결
  - 자동 변수 선택 (Feature Selection)
  - 과적합 방지
  - 모델의 해석 가능성 향상

##### Lasso 회귀 작동 원리
- **비용 함수**:
  ```
  Cost = RSS + λ∑|βⱼ|
  여기서 RSS = Σ(yᵢ - ŷᵢ)² (잔차제곱합)
       λ = 정규화 매개변수 (regularization parameter)
       |βⱼ| = 회귀계수의 절댓값 (L1 norm)
  ```

##### Lasso vs Ridge 회귀 비교
| 구분 | Lasso (L1) | Ridge (L2) |
|------|------------|------------|
| 정규화 항 | λ∑\|βⱼ\| | λ∑βⱼ² |
| 특징 | 계수를 0으로 만듦 | 계수를 0에 가깝게 축소 |
| 변수 선택 | 자동 변수 선택 | 모든 변수 유지 |
| 다중공선성 처리 | 상관 변수 중 일부 제거 | 상관 변수들의 계수 균등 분배 |

##### 다중공선성 해결 메커니즘
- **변수 선택**: 상관관계가 높은 변수들 중 일부의 계수를 0으로 만들어 제거
- **계수 축소**: 중요하지 않은 변수들의 영향력을 줄임
- **희소성**: 결과 모델이 적은 수의 변수만 사용 (sparse model)

##### λ(람다) 값의 영향
- **λ = 0**: 일반 선형 회귀와 동일
- **λ 증가**: 더 많은 변수가 제거됨 (계수가 0이 됨)
- **λ 과도하게 큰 경우**: 모든 계수가 0이 되어 과소적합 발생

##### Python 구현 예시
```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 교차 검증으로 최적 λ 찾기
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_scaled, y)
optimal_alpha = lasso_cv.alpha_

# 최적 λ로 Lasso 회귀 훈련
lasso = Lasso(alpha=optimal_alpha)
lasso.fit(X_scaled, y)

# 선택된 변수 확인
selected_features = X.columns[lasso.coef_ != 0]
print(f"선택된 변수: {selected_features.tolist()}")
print(f"제거된 변수 개수: {sum(lasso.coef_ == 0)}")
```

##### 장단점
**장점:**
- 자동 변수 선택으로 모델 단순화
- 해석 가능성 유지 (원본 변수 사용)
- 과적합 방지
- 다중공선성 문제 해결

**단점:**
- 상관관계가 높은 변수들 중 임의로 하나만 선택할 수 있음
- 그룹화된 변수들의 중요성을 놓칠 수 있음
- λ 값 선택이 중요함 (하이퍼파라미터 튜닝 필요)

##### PCA vs Lasso 선택 기준
- **PCA 선택**: 차원 축소가 주목적, 해석보다 성능 중시
- **Lasso 선택**: 변수 선택이 주목적, 해석 가능성 중시

### 1.3 다중 공선성 확인 방법

#### 1) 상관 행렬 (Correlation Matrix)
- **목적**: 독립 변수들 간의 선형 상관 관계 파악
- **계산 방법**: 
  ```
  r(X₁, X₂) = Cov(X₁, X₂) / (σ(X₁) × σ(X₂))
  또는 피어슨 상관계수: r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]
  ```
- **기준**: 일반적으로 |r| > 0.7~0.8 이면 높은 상관 관계로 판단
- **장점**: 간단하고 직관적, 시각화 용이
- **단점**: 두 변수 간 관계만 확인 가능 (다변수 관계 파악 제한)

#### 2) 허용도/공차 (Tolerance)
- **정의**: 1 - R²ᵢ (특정 변수를 다른 변수들로 회귀했을 때의 결정계수)
- **계산 방법**:
  ```
  1단계: Xᵢ = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ (Xᵢ를 제외한 모든 변수로 회귀)
  2단계: R²ᵢ 계산 (결정계수)
  3단계: Tolerance = 1 - R²ᵢ
  ```
- **기준**: Tolerance < 0.1 이면 다중 공선성 의심
- **해석**: 값이 작을수록 해당 변수가 다른 변수들에 의해 많이 설명됨

#### 3) VIF (Variance Inflation Factor) ⭐ **가장 널리 사용**
- **정의**: VIF = 1 / Tolerance
- **계산 방법**:
  ```
  1단계: Xᵢ = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ (Xᵢ를 다른 변수들로 회귀)
  2단계: R²ᵢ 계산
  3단계: VIFᵢ = 1 / (1 - R²ᵢ)
  ```
- **기준**: 
  - VIF > 10: 다중 공선성 문제 있음
  - VIF > 5: 주의 필요
  - VIF < 2: 문제 없음
- **장점**: 해석이 쉽고 임계값이 명확함
- **Python 구현**: `statsmodels.stats.outliers_influence.variance_inflation_factor`

#### 4) 조건 수 (Condition Number)
- **정의**: 설계 행렬 X의 최대 고유값과 최소 고유값의 비율의 제곱근
- **계산 방법**:
  ```
  1단계: X'X 행렬 계산 (X는 설계행렬)
  2단계: X'X의 고유값(eigenvalue) 계산: λ₁, λ₂, ..., λₖ
  3단계: 조건수 = √(λ_max / λ_min)
  ```
- **기준**: 
  - CN > 30: 다중 공선성 의심
  - CN > 100: 심각한 다중 공선성
- **특징**: 전체 데이터셋의 수치적 안정성을 평가

#### Python 코드 예시
```python
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data

# 조건수 계산
def condition_number(X):
    eigenvalues = np.linalg.eigvals(X.T @ X)
    return np.sqrt(eigenvalues.max() / eigenvalues.min())
```

### 1.4 모델 진단 방법
1. 선형성 검정 (Linearity Test) - sns.regplot
2. 잔차 정규성 검정 (Residual Normality Test) - stats.zscore
3. 독립성 검정 (Independence Test) - sm.stats.durbin_watson
4. 등분산성 검정 (Homoscedasticity Test) - stats.zscore
5. 다중공선성 검정 (Multicollinearity Test) - variance_inflation_factor

### 1.5 Linear Model과 Linear Regression의 차이

#### 1) Linear Model (선형 모델) - 포괄적 개념
- **정의**: 예측 변수들의 선형 결합으로 표현되는 모든 모델의 총칭
- **수식**: `f(x) = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ`
- **범위**: 회귀와 분류 문제 모두 포함
- **특징**: 입력 변수들과 출력 사이의 관계가 선형적

##### Linear Model에 포함되는 기법들
**회귀 문제:**
- Linear Regression (선형 회귀)
- Ridge Regression (릿지 회귀) 
- Lasso Regression (라쏘 회귀)
- Elastic Net
- Polynomial Regression (다항 회귀)

**분류 문제:**
- Logistic Regression (로지스틱 회귀)
- Linear SVM
- Linear Discriminant Analysis (LDA)

#### 2) Linear Regression (선형 회귀) - 구체적 기법
- **정의**: 연속적인 종속 변수를 예측하는 특정한 선형 모델
- **목적**: 최소제곱법을 사용하여 오차를 최소화
- **출력**: 연속적인 수치 값만
- **적용**: 회귀 문제에만 사용

#### 3) 주요 차이점 비교표

| 구분 | Linear Model | Linear Regression |
|------|-------------|------------------|
| **개념** | 포괄적 상위 개념 | 구체적 하위 기법 |
| **문제 유형** | 회귀 + 분류 | 회귀만 |
| **출력 형태** | 다양 (확률, 클래스, 수치) | 연속 수치만 |
| **손실 함수** | 다양 (MSE, Cross-entropy 등) | MSE (평균제곱오차) |
| **예시** | 로지스틱 회귀, SVM, LDA 등 | 단순/다중 선형 회귀 |

#### 4) Python 코드 예시
```python
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import LinearSVC
import numpy as np

# 샘플 데이터
X = np.random.randn(100, 3)
y_regression = np.random.randn(100)  # 연속 변수 (회귀)
y_classification = np.random.randint(0, 2, 100)  # 이진 분류

# Linear Regression (선형 회귀) - 연속 변수 예측
lr = LinearRegression()
lr.fit(X, y_regression)
print("Linear Regression 예측:", lr.predict(X[:5]))

# 다른 Linear Model들
# 1. Ridge Regression (선형 모델 + L2 정규화)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y_regression)

# 2. Lasso Regression (선형 모델 + L1 정규화)
lasso = Lasso(alpha=1.0)
lasso.fit(X, y_regression)

# 3. Logistic Regression (선형 모델 + 분류)
logistic = LogisticRegression()
logistic.fit(X, y_classification)
print("Logistic Regression 예측:", logistic.predict(X[:5]))

# 4. Linear SVM (선형 모델 + 다른 목적 함수)
svm = LinearSVC()
svm.fit(X, y_classification)
```

### 1.6 독립 변수를 갖고 종속 변수에 대해 예측
- 독립 변수의 scale의 범위가 너무 클 때 그 값을 그대로 넣고 처리하면 결과가 왜곡될 수 있다.
- 이때, -1부터 1 사이 또는 0부터 1 사이의 범위로 변환하는 **정규화(정규화/표준화) 처리 과정**이 필요하다.
- 정규화 처리를 거치면 더 정확한 처리가 가능하다.

#### 추가 설명
- **정규화(Normalization)**: 데이터의 최소값과 최대값을 이용해 0~1 범위로 변환하는 방법 (MinMaxScaler 등 사용)
- **표준화(Standardization)**: 데이터의 평균을 0, 표준편차를 1로 맞추는 방법 (StandardScaler 등 사용)
- **필요성**: 
  - 서로 다른 단위를 가진 변수들이 있을 때, 큰 값의 변수가 모델에 과도한 영향을 주는 것을 방지
  - 경사하강법 등 최적화 알고리즘의 수렴 속도 향상
  - 회귀계수 해석의 신뢰성 확보
- **실무 적용 예시**:
  ```python
  from sklearn.preprocessing import MinMaxScaler, StandardScaler

  # MinMax 정규화
  scaler = MinMaxScaler()
  X_norm = scaler.fit_transform(X)

  # 표준화
  scaler = StandardScaler()
  X_std = scaler.fit_transform(X)
  ```

### 1.7 선형회귀 모델의 평가지표

- **MAE (Mean Absolute Error, 평균 절대 오차)**
  - 실제값과 예측값의 차이의 절대값 평균
  - 직관적으로 오차의 크기를 해석할 수 있음
  - 이상치에 덜 민감함

- **MSE (Mean Squared Error, 평균 제곱 오차)**
  - 실제값과 예측값의 차이를 제곱하여 평균
  - 큰 오차에 더 큰 패널티를 부여 (이상치에 민감)
  - 모델 최적화 시 주로 사용되는 손실 함수

- **RMSE (Root Mean Squared Error, 평균 제곱근 오차)**
  - MSE에 제곱근을 취한 값
  - 실제 데이터 단위와 동일해 해석이 쉬움
  - 오차가 클수록 더 크게 반영

- **R² (결정계수, Coefficient of Determination)**
  - 모델이 실제 데이터를 얼마나 잘 설명하는지 나타내는 지표 (0~1 사이)
  - 1에 가까울수록 예측력이 높음
  - 전체 분산 중 모델이 설명하는 분산의 비율

#### Python 예시
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
```

### 1.8 인공지능 데이터셋에서의 특징(feature)와 데이터 모델 성능에서의 평가지표의 차이

- **Feature(특징, 변수)**
  - 인공지능에서 feature는 사람이 직접 해석하기 위한 것이 아니라, 모델이 데이터를 더 잘 이해하고 학습할 수 있도록 추출·가공하는 데이터의 속성이나 값이다.
  - 예시: 이미지의 픽셀값, 텍스트의 단어 빈도, 센서의 측정값 등
  - 좋은 feature를 만들면 모델의 예측 성능이 크게 향상됨 (Feature Engineering)

- **평가지표(Metric)**
  - 평가지표는 모델이 얼마나 잘 예측하는지, 얼마나 성능이 좋은지 정량적으로 비교·평가하기 위해 사용하는 기준이다.
  - 예시: MAE, MSE, RMSE, R²(회귀), Accuracy, Precision, Recall, F1-score(분류)
  - 평가지표를 통해 여러 모델의 성능을 객관적으로 비교하고, 최적의 모델을 선택할 수 있음

## 2. 학습 데이터, 검증 데이터(https://www.researchgate.net/figure/Binary-split-Scikit-learns-train-test-split-function-was-used-for-the-splits-as-it_fig19_344331692)
- **학습 데이터(Training Data)**: 모델을 학습시키기 위해 사용하는 데이터
- **검증 데이터(Validation Data)**: 학습된 모델의 성능을 평가하기 위해 사용하는 데이터
- **테스트 데이터(Test Data)**: 최종 모델의 성능을 평가하기 위해 사용하는 데이터 (학습 및 검증에 사용되지 않음)

## 3. 시계열 데이터를 섞으면(shuffle) 안 되는 이유
- 시계열 데이터는 시간의 흐름에 따라 순서가 중요한 데이터이다.
- 데이터를 섞으면 이전의 정보가 사라져 모델이 시간적 패턴을 학습할 수 없게 된다.
- 예를 들어, 주식 가격 예측 모델에서 과거 가격 정보를 섞으면 미래 가격을 예측할 수 없다.

## 4. * Linear Regression의 기본 알고리즘에 오버피팅 방지 목적의 제약조건을 담은 Ridge, Lasso, ElasticNet 회귀모형이 있다.(https://cafe.daum.net/flowlife/SBU0/27)
- **오버피팅(Overfitting)**: 모델이 학습 데이터에 너무 과도하게 적합되어, 새로운 데이터(테스트 데이터)에서는 성능이 떨어지는 현상.
- **제약조건(Regularization)**: 모델의 복잡도를 줄여 오버피팅을 방지하기 위해 회귀계수에 패널티(제약)를 추가하는 방법.

### 주요 회귀모형
- **Ridge 회귀**: L2 정규화(제곱합 패널티)를 적용하여 모든 회귀계수를 0에 가깝게 축소. 변수 선택은 하지 않지만, 다중공선성 문제를 완화.
- **Lasso 회귀**: L1 정규화(절댓값 패널티)를 적용하여 일부 회귀계수를 0으로 만들어 변수 선택 효과. 모델을 더 단순하게 만들 수 있음.
- **ElasticNet 회귀**: L1과 L2 정규화를 동시에 적용하여 Ridge와 Lasso의 장점을 결합. 변수 선택과 계수 축소를 모두 수행.

### 정규화의 효과
- 모델의 복잡도를 조절하여 과적합 방지
- 불필요한 변수의 영향력 감소 또는 제거
- 다중공선성 문제 완화
- 일반화 성능(새로운 데이터에 대한 예측력) 향상

#### Python 예시
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
```

## 5. 머신러닝에서의 L1, L2 규제

- **규제(Regularization)**: 모델이 과적합(overfitting)되는 것을 방지하기 위해, 회귀계수(가중치)에 패널티(제약)를 추가하는 방법입니다.

### L1 규제 (Lasso, L1 Regularization)
- **정의**: 회귀계수의 절댓값 합(∑|βⱼ|)에 패널티를 부여
- **효과**: 일부 계수를 0으로 만들어 변수 선택 효과(희소성, feature selection)
- **적용 모델**: Lasso 회귀, ElasticNet
- **수식**:  
  ```
  비용 함수 = RSS + λ∑|βⱼ|
  ```
  - λ: 규제 강도(하이퍼파라미터)

### L2 규제 (Ridge, L2 Regularization)
- **정의**: 회귀계수의 제곱합(∑βⱼ²)에 패널티를 부여
- **효과**: 모든 계수를 0에 가깝게 축소(변수 선택은 하지 않음)
- **적용 모델**: Ridge 회귀, ElasticNet
- **수식**:  
  ```
  비용 함수 = RSS + λ∑βⱼ²
  ```
  - λ: 규제 강도(하이퍼파라미터)

### 비교 및 특징
| 구분      | L1 규제 (Lasso)         | L2 규제 (Ridge)         |
|-----------|------------------------|-------------------------|
| 패널티    | 절댓값 합               | 제곱합                  |
| 변수 선택 | O (계수 0으로 만듦)     | X (모든 변수 유지)      |
| 희소성    | O                      | X                      |
| 다중공선성| 일부 변수 제거          | 계수 균등 분배          |

### 정리
- **L1 규제**는 변수 선택이 필요할 때, **L2 규제**는 모든 변수를 활용하되 계수를 축소하고 싶을 때 사용합니다.
- 두 규제를 혼합한 **ElasticNet**도 많이 활용됩니다.

#### Python 예시
```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)  # L2 규제
lasso = Lasso(alpha=0.1)  # L1 규제
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
```
