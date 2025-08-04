# NumPy 로그 변환, 정규화와 경사하강법 완전 가이드

## 개요

NumPy를 사용한 로그 변환, 정규화, 표준화에 대한 완전 가이드입니다. 특히 **경사하강법(Gradient Descent)**과의 연관성을 중심으로 데이터 전처리의 중요성을 다룹니다.

## 1. NumPy 출력 옵션 설정

### 1.1 기본 출력 옵션
```python
import numpy as np
import matplotlib.pyplot as plt

# 출력 옵션 설정
np.set_printoptions(suppress=True, precision=6)
# suppress=True: 과학적 표기법 억제 (1e-5 → 0.00001)
# precision=6: 소수점 이하 6자리까지 표시
```

### 1.2 다양한 출력 옵션들
```python
# 상세한 출력 옵션 설정
np.set_printoptions(
    suppress=True,        # 과학적 표기법 억제
    precision=4,          # 소수점 이하 자릿수
    threshold=10,         # 생략하지 않고 출력할 최대 요소 수
    linewidth=120,        # 한 줄 최대 문자 수
    edgeitems=3          # 배열 시작/끝에서 출력할 요소 수
)

# 출력 예시
large_array = np.random.randn(20)
print("큰 배열 출력:", large_array)

# 원래 설정으로 복원
np.set_printoptions()  # 기본값으로 복원
```

## 2. 경사하강법과 데이터 전처리의 관계

### 2.1 경사하강법이란?
```python
def gradient_descent_concept():
    """경사하강법 기본 개념"""
    print("=== 경사하강법(Gradient Descent) 개념 ===")
    print("목적: 손실 함수(Loss Function)의 최솟값을 찾는 최적화 알고리즘")
    print()
    print("기본 원리:")
    print("1. 현재 위치에서 기울기(gradient) 계산")
    print("2. 기울기 반대 방향으로 조금씩 이동")
    print("3. 최솟값에 도달할 때까지 반복")
    print()
    print("수식: θ_new = θ_old - α × ∇J(θ)")
    print("  θ: 파라미터(가중치)")
    print("  α: 학습률(learning rate)")
    print("  ∇J(θ): 손실함수의 기울기")

gradient_descent_concept()
```

### 2.2 전처리가 경사하강법에 미치는 영향
```python
def preprocessing_impact_on_gradient_descent():
    """전처리가 경사하강법에 미치는 영향 시연"""
    print("=== 전처리가 경사하강법에 미치는 영향 ===")
    
    # 1. 스케일 차이가 큰 데이터 생성
    np.random.seed(42)
    n_samples = 100
    
    # 특성 1: 나이 (20-60)
    age = np.random.uniform(20, 60, n_samples)
    # 특성 2: 연봉 (2000-8000만원) - 스케일이 매우 다름!
    salary = np.random.uniform(2000, 8000, n_samples)
    
    # 타겟 변수 (집값: 나이와 연봉에 비례)
    target = 0.5 * age + 0.001 * salary + np.random.normal(0, 2, n_samples)
    
    print("원본 데이터 스케일:")
    print(f"나이 범위: {age.min():.1f} ~ {age.max():.1f}")
    print(f"연봉 범위: {salary.min():.0f} ~ {salary.max():.0f}")
    print(f"스케일 비율: {salary.max()/age.max():.0f}:1")
    
    return age, salary, target

def demonstrate_gradient_descent_without_scaling():
    """스케일링 없는 경사하강법 문제점"""
    age, salary, target = preprocessing_impact_on_gradient_descent()
    
    # 특성 행렬 구성 (bias 포함)
    X_raw = np.column_stack([np.ones(len(age)), age, salary])
    y = target
    
    print("\n=== 스케일링 없는 경사하강법 ===")
    
    # 간단한 경사하강법 구현
    def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        
        for i in range(iterations):
            # 예측값 계산
            predictions = X.dot(theta)
            # 오차 계산
            errors = predictions - y
            # 비용 함수 (MSE)
            cost = np.sum(errors**2) / (2*m)
            cost_history.append(cost)
            
            # 기울기 계산
            gradients = X.T.dot(errors) / m
            # 파라미터 업데이트
            theta = theta - learning_rate * gradients
            
            # 비용이 발산하면 중단
            if cost > 1e10:
                print(f"  발산 발생! {i+1}번째 반복에서 중단")
                break
                
            if i % 200 == 0:
                print(f"  반복 {i}: 비용 = {cost:.6f}")
        
        return theta, cost_history
    
    # 높은 학습률로 테스트 (문제 상황 재현)
    try:
        theta_raw, costs_raw = gradient_descent(X_raw, y, learning_rate=0.01)
        print(f"  최종 파라미터: {theta_raw}")
    except:
        print("  ❌ 학습 실패: 기울기 폭발 발생!")

demonstrate_gradient_descent_without_scaling()
```

### 2.3 전처리 후 경사하강법 개선
```python
def demonstrate_gradient_descent_with_scaling():
    """스케일링 후 경사하강법 개선 효과"""
    age, salary, target = preprocessing_impact_on_gradient_descent()
    
    print("\n=== 스케일링 후 경사하강법 ===")
    
    # 특성 정규화 (Min-Max)
    age_scaled = (age - age.min()) / (age.max() - age.min())
    salary_scaled = (salary - salary.min()) / (salary.max() - salary.min())
    
    # 특성 행렬 구성 (정규화된 데이터)
    X_scaled = np.column_stack([np.ones(len(age)), age_scaled, salary_scaled])
    y = target
    
    print("정규화된 데이터 스케일:")
    print(f"나이 범위: {age_scaled.min():.3f} ~ {age_scaled.max():.3f}")
    print(f"연봉 범위: {salary_scaled.min():.3f} ~ {salary_scaled.max():.3f}")
    
    # 경사하강법 (동일한 함수 재사용)
    def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
        m, n = X.shape
        theta = np.zeros(n)
        cost_history = []
        
        for i in range(iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            cost = np.sum(errors**2) / (2*m)
            cost_history.append(cost)
            
            gradients = X.T.dot(errors) / m
            theta = theta - learning_rate * gradients
            
            if i % 200 == 0:
                print(f"  반복 {i}: 비용 = {cost:.6f}")
        
        return theta, cost_history
    
    # 동일한 학습률로 테스트
    theta_scaled, costs_scaled = gradient_descent(X_scaled, y, learning_rate=0.1)
    print(f"  ✅ 학습 성공!")
    print(f"  최종 파라미터: {theta_scaled}")
    print(f"  최종 비용: {costs_scaled[-1]:.6f}")
    
    return costs_scaled

costs = demonstrate_gradient_descent_with_scaling()
```

## 3. 로그 변환 (Logarithmic Transformation)

### 3.1 다양한 로그 함수들
```python
def demonstrate_log_functions():
    """다양한 로그 함수 시연"""
    value = 3.45
    
    print("=== 로그 함수 비교 ===")
    print(f"원본 값: {value}")
    print(f"log₂({value}) = {np.log2(value):.6f}")   # 밑이 2인 로그
    print(f"log₁₀({value}) = {np.log10(value):.6f}") # 상용로그 (밑이 10)
    print(f"ln({value}) = {np.log(value):.6f}")      # 자연로그 (밑이 e)
    
    # 로그 함수 정리
    print("\n=== 로그 함수 정리 ===")
    print("np.log2()  : 밑이 2인 로그 (컴퓨터 과학)")
    print("np.log10() : 상용로그, 밑이 10 (일반적)")
    print("np.log()   : 자연로그, 밑이 e (수학/통계)")

demonstrate_log_functions()
```

### 3.2 로그 변환과 경사하강법
```python
def log_transformation_for_gradient_descent():
    """로그 변환이 경사하강법에 미치는 영향"""
    print("=== 로그 변환과 경사하강법 ===")
    
    # 지수적으로 증가하는 데이터 (웹사이트 방문자 수)
    days = np.arange(1, 101)
    visitors = 100 * np.exp(0.05 * days) + np.random.normal(0, 100, 100)
    
    print("원본 데이터 (지수적 성장):")
    print(f"방문자 수 범위: {visitors.min():.0f} ~ {visitors.max():.0f}")
    print(f"데이터 스케일 비율: {visitors.max()/visitors.min():.1f}:1")
    
    # 로그 변환
    log_visitors = np.log(visitors)
    print(f"\n로그 변환 후:")
    print(f"로그 방문자 범위: {log_visitors.min():.3f} ~ {log_visitors.max():.3f}")
    print(f"스케일 비율: {(log_visitors.max()-log_visitors.min()):.1f} (차이)")
    
    print("\n경사하강법에서의 이점:")
    print("✅ 기울기가 안정적이 됨")
    print("✅ 학습률 설정이 쉬워짐") 
    print("✅ 수치적 안정성 향상")
    print("✅ 이상치의 영향 감소")
    
    return days, visitors, log_visitors

log_transformation_for_gradient_descent()
```

### 3.3 배열에 대한 로그 변환
```python
def log_transformation_example():
    """배열 로그 변환 예제"""
    # 원본 데이터 (다양한 크기의 값들)
    values = np.array([3.45, 34.5, 0.01, 10, 100, 1000])
    print("=== 로그 변환 예제 ===")
    print(f"원본 자료: {values}")
    print(f"원본 범위: {values.min():.2f} ~ {values.max():.2f}")
    
    # 상용로그 변환
    log10_values = np.log10(values)
    print(f"상용로그 변환: {log10_values}")
    print(f"로그 변환 범위: {log10_values.min():.2f} ~ {log10_values.max():.2f}")
    
    # 자연로그 변환
    ln_values = np.log(values)  # 주의: np.log는 자연로그!
    print(f"자연로그 변환: {ln_values}")
    print(f"자연로그 범위: {ln_values.min():.2f} ~ {ln_values.max():.2f}")
    
    print(f"배열 형태: {log10_values.shape}")
    
    return values, log10_values, ln_values

values, log10_vals, ln_vals = log_transformation_example()
```

## 4. 정규화 (Normalization)

### 4.1 Min-Max 정규화와 경사하강법
```python
def min_max_normalization_for_gradient_descent(data):
    """Min-Max 정규화와 경사하강법 관계"""
    print("=== Min-Max 정규화와 경사하강법 ===")
    
    # 공식: (x - min) / (max - min)
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    
    print(f"원본 데이터: {data}")
    print(f"원본 범위: [{min_val:.2f}, {max_val:.2f}]")
    print(f"정규화 결과: {normalized}")
    print(f"정규화 범위: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    print("\n경사하강법에서의 이점:")
    print("✅ 모든 특성이 동일한 스케일 (0~1)")
    print("✅ 기울기 크기가 균형적")
    print("✅ 학습률 설정이 용이")
    print("✅ 빠른 수렴")
    
    return normalized

# 로그 변환된 데이터를 정규화
normalized_log = min_max_normalization_for_gradient_descent(log10_vals)
```

### 4.2 다양한 범위로 정규화
```python
def custom_range_normalization(data, target_min=0, target_max=1):
    """사용자 정의 범위로 정규화"""
    # 공식: target_min + (x - min) * (target_max - target_min) / (max - min)
    min_val = np.min(data)
    max_val = np.max(data)
    
    normalized = target_min + (data - min_val) * (target_max - target_min) / (max_val - min_val)
    
    print(f"=== 사용자 정의 범위 정규화 [{target_min}, {target_max}] ===")
    print(f"정규화 결과: {normalized}")
    
    return normalized

# -1 ~ 1 범위로 정규화 (tanh 활성화 함수에 적합)
normalized_custom = custom_range_normalization(values, -1, 1)
```

## 5. 표준화 (Standardization/Z-score Normalization)

### 5.1 Z-score 표준화와 경사하강법
```python
def z_score_standardization_for_gradient_descent(data):
    """Z-score 표준화와 경사하강법 관계"""
    print("=== Z-score 표준화와 경사하강법 ===")
    
    # 공식: (x - 평균) / 표준편차
    mean_val = np.mean(data)
    std_val = np.std(data)
    standardized = (data - mean_val) / std_val
    
    print(f"원본 데이터: {data}")
    print(f"원본 평균: {mean_val:.2f}, 표준편차: {std_val:.2f}")
    print(f"표준화 결과: {standardized}")
    print(f"표준화 평균: {np.mean(standardized):.6f}")
    print(f"표준화 표준편차: {np.std(standardized):.6f}")
    
    print("\n경사하강법에서의 이점:")
    print("✅ 평균이 0, 표준편차가 1")
    print("✅ 정규분포를 따르는 데이터에 최적")
    print("✅ 가중치 초기화와 잘 맞음")
    print("✅ 배치 정규화와 유사한 효과")
    
    return standardized

# 원본 데이터 표준화
standardized_data = z_score_standardization_for_gradient_descent(values)
```

### 5.2 로버스트 표준화
```python
def robust_standardization(data):
    """로버스트 표준화: 중앙값과 IQR 사용 (이상치에 강함)"""
    print("=== 로버스트 표준화 ===")
    
    # 공식: (x - 중앙값) / IQR
    median_val = np.median(data)
    q75 = np.percentile(data, 75)
    q25 = np.percentile(data, 25)
    iqr = q75 - q25
    
    robust_scaled = (data - median_val) / iqr
    
    print(f"중앙값: {median_val:.2f}")
    print(f"IQR (Q3-Q1): {iqr:.2f}")
    print(f"로버스트 표준화 결과: {robust_scaled}")
    
    print("\n경사하강법에서의 이점:")
    print("✅ 이상치에 강함")
    print("✅ 안정적인 기울기")
    print("✅ 비정규분포 데이터에 적합")
    
    return robust_scaled

# 이상치가 있는 데이터로 테스트
data_with_outliers = np.array([1, 2, 3, 4, 5, 100])  # 100이 이상치
robust_scaled = robust_standardization(data_with_outliers)
```

## 6. 경사하강법을 위한 전처리 전략

### 6.1 경사하강법 최적화를 위한 전처리 선택
```python
def preprocessing_strategy_for_gradient_descent():
    """경사하강법을 위한 전처리 전략"""
    print("=== 경사하강법을 위한 전처리 전략 ===")
    
    print("1. 특성 스케일 분석:")
    print("   📊 모든 특성의 범위와 분포 확인")
    print("   📊 상관관계 분석")
    print("   📊 이상치 탐지")
    
    print("\n2. 전처리 방법 선택:")
    print("   🔄 스케일 차이 > 100배 → 로그 변환 고려")
    print("   🔄 정규분포 → Z-score 표준화")
    print("   🔄 균등분포 → Min-Max 정규화")
    print("   🔄 이상치 많음 → Robust Scaling")
    
    print("\n3. 경사하강법 관점에서:")
    print("   ⚡ 빠른 수렴: Min-Max (0~1)")
    print("   ⚡ 안정적 학습: Z-score")
    print("   ⚡ 이상치 대응: Robust")
    print("   ⚡ 지수적 데이터: Log + Standard")

preprocessing_strategy_for_gradient_descent()
```

### 6.2 전처리별 학습률 설정 가이드
```python
def learning_rate_guide_by_preprocessing():
    """전처리 방법별 학습률 설정 가이드"""
    print("=== 전처리별 학습률 설정 가이드 ===")
    
    preprocessing_methods = {
        "원본 데이터 (스케일 차이 큼)": {
            "학습률": "0.00001 ~ 0.0001",
            "문제점": "수렴 느림, 발산 위험",
            "권장": "전처리 필수"
        },
        "Min-Max 정규화 (0~1)": {
            "학습률": "0.01 ~ 0.1",
            "장점": "안정적, 빠른 수렴",
            "권장": "신경망, 거리 기반"
        },
        "Z-score 표준화": {
            "학습률": "0.001 ~ 0.01",
            "장점": "이론적 근거 강함",
            "권장": "선형 모델, 정규분포"
        },
        "로그 + 표준화": {
            "학습률": "0.001 ~ 0.01",
            "장점": "왜곡 분포 처리",
            "권장": "지수적 성장 데이터"
        }
    }
    
    for method, info in preprocessing_methods.items():
        print(f"\n{method}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

learning_rate_guide_by_preprocessing()
```

### 6.3 실제 경사하강법 구현 예제
```python
def complete_gradient_descent_example():
    """완전한 경사하강법 구현 예제"""
    print("=== 완전한 경사하강법 예제 ===")
    
    # 가상의 데이터 생성 (집값 예측)
    np.random.seed(42)
    n_samples = 200
    
    # 다양한 스케일의 특성들
    size = np.random.uniform(20, 200, n_samples)      # 평방미터 (20-200)
    age = np.random.uniform(1, 50, n_samples)         # 연식 (1-50년)
    distance = np.random.uniform(0.1, 20, n_samples)  # 거리 (0.1-20km)
    
    # 실제 관계식 (집값 = 크기*2 - 연식*0.5 - 거리*3 + 노이즈)
    true_price = size * 2 - age * 0.5 - distance * 3 + np.random.normal(0, 10, n_samples)
    
    print("원본 데이터 스케일:")
    print(f"크기: {size.min():.1f} ~ {size.max():.1f}")
    print(f"연식: {age.min():.1f} ~ {age.max():.1f}")  
    print(f"거리: {distance.min():.1f} ~ {distance.max():.1f}")
    
    # 전처리: Min-Max 정규화
    def normalize_feature(feature):
        return (feature - feature.min()) / (feature.max() - feature.min())
    
    size_norm = normalize_feature(size)
    age_norm = normalize_feature(age)
    distance_norm = normalize_feature(distance)
    
    # 특성 행렬 구성 (bias 포함)
    X = np.column_stack([np.ones(n_samples), size_norm, age_norm, distance_norm])
    y = true_price
    
    # 경사하강법 구현
    def gradient_descent_with_history(X, y, learning_rate=0.1, iterations=1000):
        m, n = X.shape
        theta = np.random.normal(0, 0.01, n)  # 작은 랜덤 초기화
        
        cost_history = []
        theta_history = []
        
        for i in range(iterations):
            # 순전파
            predictions = X.dot(theta)
            errors = predictions - y
            cost = np.sum(errors**2) / (2*m)
            
            # 역전파 (기울기 계산)
            gradients = X.T.dot(errors) / m
            
            # 파라미터 업데이트
            theta = theta - learning_rate * gradients
            
            # 기록
            cost_history.append(cost)
            theta_history.append(theta.copy())
            
            if i % 100 == 0:
                print(f"반복 {i:4d}: 비용 = {cost:10.2f}, 기울기 크기 = {np.linalg.norm(gradients):.6f}")
        
        return theta, cost_history, theta_history
    
    # 학습 실행
    final_theta, costs, theta_hist = gradient_descent_with_history(X, y)
    
    print(f"\n최종 파라미터: {final_theta}")
    print(f"최종 비용: {costs[-1]:.2f}")
    print(f"수렴 여부: {'✅ 수렴' if costs[-1] < costs[10] else '❌ 발산'}")
    
    return X, y, final_theta, costs

# 실행
X, y, theta, costs = complete_gradient_descent_example()
```

## 7. 로그 변환의 역변환 (Inverse Transformation) 완전 가이드

### 7.1 역변환의 기본 개념
```python
def log_inverse_transformation_basics():
    """로그 변환 역변환 기본 개념"""
    print("=== 로그 변환 역변환 기본 개념 ===")
    
    # 원본 데이터
    original_data = np.array([1, 10, 100, 1000, 10000])
    print(f"원본 데이터: {original_data}")
    
    # 로그 변환 (상용로그)
    log_transformed = np.log10(original_data)
    print(f"로그10 변환: {log_transformed}")
    
    # 역변환 (10의 거듭제곱)
    inverse_transformed = 10 ** log_transformed
    # 또는 np.power(10, log_transformed)
    print(f"역변환 결과: {inverse_transformed}")
    
    # 정확성 검증
    is_equal = np.allclose(original_data, inverse_transformed)
    print(f"원본과 역변환 결과 일치: {is_equal}")
    
    print("\n=== 다양한 로그 함수의 역변환 ===")
    value = 100
    
    # 자연로그와 역변환
    ln_val = np.log(value)
    exp_val = np.exp(ln_val)
    print(f"자연로그: ln({value}) = {ln_val:.6f}")
    print(f"지수함수: e^{ln_val:.6f} = {exp_val:.6f}")
    
    # 상용로그와 역변환  
    log10_val = np.log10(value)
    power10_val = 10 ** log10_val
    print(f"상용로그: log10({value}) = {log10_val:.6f}")
    print(f"거듭제곱: 10^{log10_val:.6f} = {power10_val:.6f}")
    
    # 밑이 2인 로그와 역변환
    log2_val = np.log2(value)
    power2_val = 2 ** log2_val
    print(f"로그2: log2({value}) = {log2_val:.6f}")
    print(f"거듭제곱: 2^{log2_val:.6f} = {power2_val:.6f}")

log_inverse_transformation_basics()
```

### 7.2 0값과 음수 처리를 위한 오프셋 역변환
```python
def offset_inverse_transformation():
    """오프셋을 사용한 로그 변환과 역변환"""
    print("=== 오프셋 로그 변환과 역변환 ===")
    
    # 0과 음수가 포함된 데이터
    problematic_data = np.array([0, 0.1, 1, 10, 100, -5])
    print(f"문제가 있는 원본 데이터: {problematic_data}")
    
    # 방법 1: 단순 오프셋 추가
    offset = 10  # 최솟값보다 큰 양수
    offset_data = problematic_data + offset
    print(f"오프셋 추가 ({offset}): {offset_data}")
    
    # 로그 변환
    log_offset = np.log(offset_data)
    print(f"로그 변환: {log_offset}")
    
    # 역변환
    exp_result = np.exp(log_offset)
    original_recovered = exp_result - offset
    print(f"역변환 (exp): {exp_result}")
    print(f"오프셋 제거: {original_recovered}")
    print(f"원본 복원 정확도: {np.allclose(problematic_data, original_recovered)}")
    
    print("\n방법 2: np.log1p와 np.expm1 사용 (더 정확)")
    # log1p(x) = log(1+x), expm1(x) = exp(x)-1
    positive_data = np.array([0, 0.1, 1, 10, 100])  # 양수만
    
    log1p_result = np.log1p(positive_data)
    expm1_result = np.expm1(log1p_result)
    
    print(f"원본: {positive_data}")
    print(f"log1p: {log1p_result}")
    print(f"expm1: {expm1_result}")
    print(f"정확도: {np.allclose(positive_data, expm1_result)}")

offset_inverse_transformation()
```

### 7.3 완전한 로그 변환 클래스 구현
```python
class LogTransformer:
    """완전한 로그 변환 및 역변환 클래스"""
    
    def __init__(self, method='log10', offset='auto'):
        """
        Parameters:
        -----------
        method : str
            'log10', 'ln', 'log2', 'log1p' 중 선택
        offset : float or 'auto'
            오프셋 값, 'auto'면 자동 계산
        """
        self.method = method
        self.offset = offset
        self.fitted_offset = None
        self.is_fitted = False
        
    def fit(self, X):
        """변환 파라미터 학습"""
        X = np.array(X)
        
        if self.method == 'log1p':
            # log1p는 오프셋이 필요 없음
            self.fitted_offset = 0
        else:
            if self.offset == 'auto':
                min_val = np.min(X)
                if min_val <= 0:
                    # 최솟값이 0 이하면 적절한 오프셋 설정
                    self.fitted_offset = abs(min_val) + 1
                else:
                    self.fitted_offset = 0
            else:
                self.fitted_offset = self.offset
                
        self.is_fitted = True
        print(f"LogTransformer 학습 완료 - 방법: {self.method}, 오프셋: {self.fitted_offset}")
        return self
    
    def transform(self, X):
        """로그 변환 수행"""
        if not self.is_fitted:
            raise ValueError("먼저 fit()을 호출해야 합니다.")
            
        X = np.array(X)
        X_shifted = X + self.fitted_offset
        
        if self.method == 'log10':
            return np.log10(X_shifted)
        elif self.method == 'ln':
            return np.log(X_shifted)
        elif self.method == 'log2':
            return np.log2(X_shifted)
        elif self.method == 'log1p':
            return np.log1p(X)  # log1p는 오프셋 불필요
        else:
            raise ValueError(f"지원하지 않는 방법: {self.method}")
    
    def inverse_transform(self, X_transformed):
        """역변환 수행"""
        if not self.is_fitted:
            raise ValueError("먼저 fit()을 호출해야 합니다.")
            
        X_transformed = np.array(X_transformed)
        
        if self.method == 'log10':
            X_exp = 10 ** X_transformed
        elif self.method == 'ln':
            X_exp = np.exp(X_transformed)
        elif self.method == 'log2':
            X_exp = 2 ** X_transformed
        elif self.method == 'log1p':
            return np.expm1(X_transformed)  # expm1은 오프셋 불필요
        else:
            raise ValueError(f"지원하지 않는 방법: {self.method}")
            
        return X_exp - self.fitted_offset
    
    def fit_transform(self, X):
        """학습과 변환을 동시에 수행"""
        return self.fit(X).transform(X)

def test_log_transformer():
    """LogTransformer 테스트"""
    print("=== LogTransformer 클래스 테스트 ===")
    
    # 다양한 데이터로 테스트
    test_cases = {
        "양수만": np.array([1, 10, 100, 1000]),
        "0 포함": np.array([0, 1, 10, 100]),
        "음수 포함": np.array([-5, 0, 5, 50, 500]),
        "소수점": np.array([0.01, 0.1, 1.5, 15.7, 157.3])
    }
    
    methods = ['log10', 'ln', 'log1p']
    
    for case_name, data in test_cases.items():
        print(f"\n📊 {case_name}: {data}")
        
        for method in methods:
            try:
                # 변환기 생성 및 학습
                transformer = LogTransformer(method=method, offset='auto')
                
                # 변환
                transformed = transformer.fit_transform(data)
                
                # 역변환
                recovered = transformer.inverse_transform(transformed)
                
                # 정확도 검증
                accuracy = np.allclose(data, recovered, rtol=1e-10)
                
                print(f"  {method:6s}: 변환 범위 {transformed.min():.3f}~{transformed.max():.3f}, "
                      f"복원 정확도 {'✅' if accuracy else '❌'}")
                
                if not accuracy:
                    max_error = np.max(np.abs(data - recovered))
                    print(f"          최대 오차: {max_error:.2e}")
                    
            except Exception as e:
                print(f"  {method:6s}: ❌ 오류 - {str(e)}")

test_log_transformer()
```

### 7.4 머신러닝에서 역변환의 실제 활용
```python
def ml_inverse_transformation_example():
    """머신러닝에서 역변환 활용 예제"""
    print("=== 머신러닝에서 역변환 활용 ===")
    
    # 집값 예측 문제 (타겟이 로그 변환된 경우)
    np.random.seed(42)
    
    # 실제 집값 (백만원 단위, 넓은 범위)
    actual_prices = np.array([
        150, 280, 450, 680, 920, 1200, 1800, 2500, 3200, 4500,
        6000, 8500, 12000, 18000, 25000, 35000, 50000
    ])
    
    print("1. 원본 집값 데이터:")
    print(f"   범위: {actual_prices.min():,} ~ {actual_prices.max():,} 만원")
    print(f"   스케일 비율: {actual_prices.max()/actual_prices.min():.1f}:1")
    
    # 로그 변환 (경사하강법을 위해)
    log_transformer = LogTransformer(method='log10', offset='auto')
    log_prices = log_transformer.fit_transform(actual_prices)
    
    print(f"\n2. 로그 변환 후:")
    print(f"   로그 범위: {log_prices.min():.3f} ~ {log_prices.max():.3f}")
    print(f"   로그 스케일 차이: {log_prices.max() - log_prices.min():.3f}")
    
    # 가상의 모델 예측 (약간의 노이즈 추가)
    predicted_log_prices = log_prices + np.random.normal(0, 0.05, len(log_prices))
    
    print(f"\n3. 모델 예측 (로그 공간):")
    print(f"   예측 로그 범위: {predicted_log_prices.min():.3f} ~ {predicted_log_prices.max():.3f}")
    
    # 역변환으로 실제 가격 복원
    predicted_actual_prices = log_transformer.inverse_transform(predicted_log_prices)
    
    print(f"\n4. 역변환된 예측 가격:")
    print(f"   예측 범위: {predicted_actual_prices.min():,.0f} ~ {predicted_actual_prices.max():,.0f} 만원")
    
    # 성능 평가 (실제 스케일에서)
    mae_actual = np.mean(np.abs(actual_prices - predicted_actual_prices))
    mape = np.mean(np.abs((actual_prices - predicted_actual_prices) / actual_prices)) * 100
    
    print(f"\n5. 모델 성능 (실제 스케일):")
    print(f"   평균 절대 오차 (MAE): {mae_actual:,.0f} 만원")
    print(f"   평균 절대 백분율 오차 (MAPE): {mape:.2f}%")
    
    # 로그 스케일에서의 성능과 비교
    mae_log = np.mean(np.abs(log_prices - predicted_log_prices))
    print(f"   로그 스케일 MAE: {mae_log:.4f}")
    
    print(f"\n6. 역변환의 중요성:")
    print(f"   ✅ 실제 비즈니스 의미로 해석 가능")
    print(f"   ✅ 실제 스케일에서 성능 평가")
    print(f"   ✅ 의사결정에 직접 활용 가능")

ml_inverse_transformation_example()
```

### 7.5 경사하강법에서 역변환의 중요성
```python
def gradient_descent_with_inverse_transformation():
    """경사하강법에서 역변환 활용 예제"""
    print("=== 경사하강법과 역변환 ===")
    
    # 지수적 성장 데이터 생성 (웹사이트 트래픽)
    days = np.arange(1, 101)
    actual_traffic = 100 * np.exp(0.05 * days) + np.random.normal(0, 200, 100)
    actual_traffic = np.maximum(actual_traffic, 1)  # 최소값 1로 제한
    
    print("1. 원본 트래픽 데이터:")
    print(f"   범위: {actual_traffic.min():.0f} ~ {actual_traffic.max():.0f} 방문자")
    print(f"   평균: {actual_traffic.mean():.0f}, 표준편차: {actual_traffic.std():.0f}")
    
    # 로그 변환
    log_transformer = LogTransformer(method='ln', offset='auto')
    log_traffic = log_transformer.fit_transform(actual_traffic)
    
    print(f"\n2. 로그 변환 후:")
    print(f"   로그 범위: {log_traffic.min():.3f} ~ {log_traffic.max():.3f}")
    print(f"   로그 평균: {log_traffic.mean():.3f}, 표준편차: {log_traffic.std():.3f}")
    
    # 간단한 선형 회귀 (경사하강법으로)
    X = np.column_stack([np.ones(len(days)), days])  # [1, day]
    y_log = log_traffic
    
    def simple_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
        theta = np.zeros(X.shape[1])
        costs = []
        
        for i in range(iterations):
            predictions = X.dot(theta)
            errors = predictions - y
            cost = np.sum(errors**2) / (2*len(y))
            costs.append(cost)
            
            gradients = X.T.dot(errors) / len(y)
            theta = theta - learning_rate * gradients
            
            if i % 200 == 0:
                print(f"   반복 {i}: 로그 스케일 비용 = {cost:.6f}")
        
        return theta, costs
    
    print(f"\n3. 경사하강법 학습 (로그 스케일):")
    theta, costs = simple_gradient_descent(X, y_log, learning_rate=0.001)
    
    # 로그 스케일에서 예측
    log_predictions = X.dot(theta)
    
    # 역변환으로 실제 스케일 예측
    actual_predictions = log_transformer.inverse_transform(log_predictions)
    
    print(f"\n4. 결과 비교:")
    print(f"   로그 스케일 MSE: {np.mean((log_traffic - log_predictions)**2):.6f}")
    print(f"   실제 스케일 MSE: {np.mean((actual_traffic - actual_predictions)**2):,.0f}")
    print(f"   실제 스케일 MAPE: {np.mean(np.abs((actual_traffic - actual_predictions)/actual_traffic))*100:.2f}%")
    
    print(f"\n5. 역변환의 가치:")
    print(f"   📈 실제 트래픽으로 결과 해석")
    print(f"   📊 비즈니스 지표로 성능 측정") 
    print(f"   🎯 실제 목표치와 비교 가능")
    
    return actual_traffic, actual_predictions, log_transformer

traffic, predictions, transformer = gradient_descent_with_inverse_transformation()
```

### 7.6 역변환 시 주의사항과 베스트 프랙티스
```python
def inverse_transformation_best_practices():
    """역변환 베스트 프랙티스"""
    print("=== 역변환 베스트 프랙티스 ===")
    
    print("1. 🔍 수치적 정확도 관리:")
    
    # 부동소수점 정밀도 문제
    original = np.array([1e-10, 1e-5, 1, 1e5, 1e10])
    
    # 정밀도가 낮은 변환
    log_vals = np.log10(original)
    recovered_low = 10 ** log_vals
    
    # 정밀도가 높은 변환 (log1p/expm1 사용)
    adjusted_original = original - 1  # log1p를 위해 조정
    log1p_vals = np.log1p(adjusted_original)
    recovered_high = np.expm1(log1p_vals) + 1
    
    print(f"   원본: {original}")
    print(f"   낮은 정밀도 복원: {recovered_low}")
    print(f"   높은 정밀도 복원: {recovered_high}")
    print(f"   낮은 정밀도 정확도: {np.allclose(original, recovered_low)}")
    print(f"   높은 정밀도 정확도: {np.allclose(original, recovered_high)}")
    
    print(f"\n2. ⚠️  오버플로우/언더플로우 방지:")
    
    # 큰 로그 값들
    large_log_values = np.array([-50, -10, 0, 10, 50, 100])
    
    try:
        # 직접 지수 계산 (오버플로우 위험)
        direct_exp = np.exp(large_log_values)
        print(f"   직접 exp(): {direct_exp}")
    except:
        print(f"   직접 exp(): 오버플로우 발생")
    
    # 안전한 방법: 클리핑
    clipped_log = np.clip(large_log_values, -50, 50)
    safe_exp = np.exp(clipped_log)
    print(f"   클리핑 후 exp(): {safe_exp}")
    
    print(f"\n3. 📝 변환 파라미터 저장:")
    
    class SafeLogTransformer:
        def __init__(self):
            self.transform_params = {}
            
        def fit_transform_with_save(self, data, method='log10'):
            # 변환 파라미터 저장
            if method == 'log10':
                min_val = np.min(data)
                offset = max(0, -min_val + 1e-8) if min_val <= 0 else 0
                
                self.transform_params = {
                    'method': method,
                    'offset': offset,
                    'original_min': min_val,
                    'original_max': np.max(data)
                }
                
                transformed = np.log10(data + offset)
                
            return transformed
        
        def inverse_transform_with_params(self, transformed_data):
            params = self.transform_params
            
            if params['method'] == 'log10':
                recovered = 10 ** transformed_data - params['offset']
                
            return recovered
        
        def get_transform_info(self):
            return self.transform_params
    
    # 사용 예제
    test_data = np.array([0.001, 0.1, 1, 10, 100])
    transformer = SafeLogTransformer()
    
    transformed = transformer.fit_transform_with_save(test_data)
    recovered = transformer.inverse_transform_with_params(transformed)
    
    print(f"   원본: {test_data}")
    print(f"   변환: {transformed}")
    print(f"   복원: {recovered}")
    print(f"   변환 정보: {transformer.get_transform_info()}")
    
    print(f"\n4. 🚫 흔한 실수들:")
    print(f"   ❌ 변환 파라미터를 저장하지 않음")
    print(f"   ❌ 테스트 데이터에 다른 변환 적용")
    print(f"   ❌ 오버플로우/언더플로우 무시")
    print(f"   ❌ 부동소수점 정밀도 문제 간과")
    
    print(f"\n5. ✅ 베스트 프랙티스:")
    print(f"   ✅ 변환 파라미터 항상 저장")
    print(f"   ✅ 안전한 수치 범위 유지")
    print(f"   ✅ 정확도 검증 포함")
    print(f"   ✅ 예외 처리 구현")

inverse_transformation_best_practices()
```

## 8. 정규화 vs 표준화 vs 로그변환 비교

### 8.1 경사하강법 관점에서 종합 비교
```python
def comprehensive_comparison_for_gradient_descent():
    """경사하강법 관점에서 전처리 방법 종합 비교"""
    print("=== 경사하강법을 위한 전처리 방법 비교 ===")
    
    # 테스트 데이터 (다양한 특성을 가진 데이터)
    np.random.seed(42)
    
    # 특성들 (다양한 분포와 스케일)
    normal_data = np.random.normal(50, 15, 100)           # 정규분포
    uniform_data = np.random.uniform(0, 100, 100)         # 균등분포  
    exponential_data = np.random.exponential(10, 100)     # 지수분포
    heavy_tail = np.random.pareto(1, 100) * 10           # 파레토분포 (heavy tail)
    
    datasets = {
        "정규분포": normal_data,
        "균등분포": uniform_data, 
        "지수분포": exponential_data,
        "긴 꼬리 분포": heavy_tail
    }
    
    methods = {
        "Min-Max": lambda x: (x - x.min()) / (x.max() - x.min()),
        "Z-score": lambda x: (x - x.mean()) / x.std(),
        "Robust": lambda x: (x - np.median(x)) / (np.percentile(x, 75) - np.percentile(x, 25)),
        "Log+Z-score": lambda x: (np.log1p(x) - np.log1p(x).mean()) / np.log1p(x).std()
    }
    
    print("데이터별 전처리 결과 비교:")
    print("-" * 80)
    
    for data_name, data in datasets.items():
        print(f"\n📊 {data_name} (원본 범위: {data.min():.1f}~{data.max():.1f})")
        
        for method_name, method_func in methods.items():
            try:
                processed = method_func(data)
                range_str = f"{processed.min():.2f}~{processed.max():.2f}"
                std_str = f"σ={processed.std():.2f}"
                print(f"  {method_name:12s}: 범위={range_str:12s}, {std_str}")
            except:
                print(f"  {method_name:12s}: ❌ 변환 실패")

comprehensive_comparison_for_gradient_descent()
```

## 9. 완전한 전처리 파이프라인

### 9.1 역변환 기능이 포함된 전처리 파이프라인
```python
class ComprehensivePreprocessor:
    """역변환 기능이 포함된 완전한 전처리 파이프라인"""
    
    def __init__(self):
        self.transformers = {}
        self.feature_names = []
        self.is_fitted = False
    
    def fit_transform(self, X, feature_names=None, methods=None):
        """데이터 학습 및 변환"""
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        if methods is None:
            methods = ['minmax'] * X.shape[1]
            
        self.feature_names = feature_names
        transformed_X = np.zeros_like(X, dtype=float)
        
        for i, (