# NumPy 로그 변환과 정규화 완전 가이드

## 개요

NumPy를 사용한 로그 변환, 정규화, 표준화에 대한 완전 가이드입니다. 데이터 전처리에서 필수적인 기법들을 다룹니다.

로그 변환은 편차가 큰 데이터를 처리할 때 필수적인 기법으로, 데이터의 분포를 개선하고 큰 범위의 차이를 줄이며, 모델이 보다 안정적으로 학습할 수 있도록 도와줍니다.

---

## 1. NumPy 출력 옵션 설정

### 1.1 기본 출력 옵션

```python
import numpy as np

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

---

## 2. 로그 변환 (Logarithmic Transformation)

### 2.1 다양한 로그 함수들

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

### 2.2 배열에 대한 로그 변환

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

### 2.3 로그 변환의 활용 목적

```python
def why_log_transformation():
    """로그 변환을 사용하는 이유"""
    print("=== 로그 변환의 목적 ===")
    
    # 1. 스케일 차이가 큰 데이터
    original_data = np.array([1, 10, 100, 1000, 10000])
    log_data = np.log10(original_data)
    
    print("1. 스케일 차이 완화:")
    print(f"   원본: {original_data}")
    print(f"   로그: {log_data}")
    print(f"   원본 범위: {original_data.max()/original_data.min():.0f}배 차이")
    print(f"   로그 범위: {log_data.max()-log_data.min():.1f} 차이")
    
    # 2. 왜곡된 분포 정규화
    skewed_data = np.random.exponential(2, 1000)  # 지수분포 (왜곡됨)
    log_skewed = np.log(skewed_data + 1)  # +1은 0값 처리용
    
    print(f"\n2. 분포 왜곡 완화:")
    print(f"   원본 왜도: {np.mean(((skewed_data - np.mean(skewed_data))/np.std(skewed_data))**3):.2f}")
    print(f"   로그 왜도: {np.mean(((log_skewed - np.mean(log_skewed))/np.std(log_skewed))**3):.2f}")
    print("   (왜도가 0에 가까울수록 정규분포에 가까움)")

why_log_transformation()
```

---

## 3. 정규화 (Normalization)

### 3.1 Min-Max 정규화 (0~1 범위)

```python
def min_max_normalization(data):
    """Min-Max 정규화: 데이터를 0~1 범위로 변환"""
    print("=== Min-Max 정규화 (0~1) ===")
    
    # 공식: (x - min) / (max - min)
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    
    print(f"원본 데이터: {data}")
    print(f"원본 범위: [{min_val:.2f}, {max_val:.2f}]")
    print(f"정규화 결과: {normalized}")
    print(f"정규화 범위: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    return normalized

# 로그 변환된 데이터를 정규화
normalized_log = min_max_normalization(log10_vals)
```

### 3.2 다양한 범위로 정규화

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

# -1 ~ 1 범위로 정규화
normalized_custom = custom_range_normalization(values, -1, 1)
```

---

## 4. 표준화 (Standardization/Z-score Normalization)

### 4.1 Z-score 표준화

```python
def z_score_standardization(data):
    """Z-score 표준화: 평균 0, 표준편차 1로 변환"""
    print("=== Z-score 표준화 ===")
    
    # 공식: (x - 평균) / 표준편차
    mean_val = np.mean(data)
    std_val = np.std(data)
    standardized = (data - mean_val) / std_val
    
    print(f"원본 데이터: {data}")
    print(f"원본 평균: {mean_val:.2f}, 표준편차: {std_val:.2f}")
    print(f"표준화 결과: {standardized}")
    print(f"표준화 평균: {np.mean(standardized):.6f}")
    print(f"표준화 표준편차: {np.std(standardized):.6f}")
    
    return standardized

# 원본 데이터 표준화
standardized_data = z_score_standardization(values)
```

### 4.2 로버스트 표준화 (Robust Scaling)

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
    
    return robust_scaled

# 이상치가 있는 데이터로 테스트
data_with_outliers = np.array([1, 2, 3, 4, 5, 100])  # 100이 이상치
robust_scaled = robust_standardization(data_with_outliers)
```

---

## 5. 정규화 vs 표준화 비교

### 5.1 언제 어떤 방법을 사용할까?

```python
def normalization_vs_standardization():
    """정규화와 표준화 비교"""
    print("=== 정규화 vs 표준화 비교 ===")
    
    # 테스트 데이터
    data = np.array([10, 20, 30, 40, 50, 100, 200])
    
    # 정규화 (0~1)
    normalized = (data - data.min()) / (data.max() - data.min())
    
    # 표준화 (Z-score)
    standardized = (data - data.mean()) / data.std()
    
    print(f"원본 데이터:     {data}")
    print(f"정규화 (0~1):    {normalized}")
    print(f"표준화 (Z-score): {standardized}")
    
    print("\n=== 사용 시나리오 ===")
    print("정규화 (Min-Max):")
    print("  ✅ 데이터 범위가 알려져 있을 때")
    print("  ✅ 신경망, 이미지 처리")
    print("  ✅ 거리 기반 알고리즘 (KNN, K-means)")
    print("  ❌ 이상치에 민감함")
    
    print("\n표준화 (Z-score):")
    print("  ✅ 데이터가 정규분포를 따를 때")
    print("  ✅ 선형 회귀, PCA")
    print("  ✅ 평균과 표준편차가 의미있을 때")
    print("  ❌ 데이터 범위를 0~1로 제한하지 않음")

normalization_vs_standardization()
```

### 5.2 실제 머신러닝에서의 활용

```python
def ml_preprocessing_example():
    """머신러닝 전처리 예제"""
    print("=== 머신러닝 전처리 파이프라인 ===")
    
    # 가상의 특성 데이터 (키, 몸무게, 나이, 연봉)
    features = np.array([
        [170, 65, 25, 30000],    # 사람 1
        [180, 80, 30, 50000],    # 사람 2  
        [160, 55, 22, 25000],    # 사람 3
        [175, 70, 35, 80000],    # 사람 4
        [185, 90, 40, 120000]    # 사람 5
    ])
    
    feature_names = ['키(cm)', '몸무게(kg)', '나이', '연봉(만원)']
    
    print("원본 데이터:")
    for i, name in enumerate(feature_names):
        print(f"{name}: {features[:, i]}")
    
    # 각 특성별로 정규화/표준화 적용
    processed_features = np.zeros_like(features, dtype=float)
    
    for i, name in enumerate(feature_names):
        if '연봉' in name:
            # 연봉은 로그 변환 후 표준화 (스케일 차이가 큼)
            log_data = np.log10(features[:, i])
            processed_features[:, i] = (log_data - log_data.mean()) / log_data.std()
            print(f"\n{name}: 로그 변환 + 표준화")
        else:
            # 나머지는 정규화 (0~1)
            col_data = features[:, i]
            processed_features[:, i] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
            print(f"\n{name}: Min-Max 정규화")
        
        print(f"  전처리 결과: {processed_features[:, i]}")

ml_preprocessing_example()
```

---

## 6. 실무 활용 팁

### 6.1 주의사항과 베스트 프랙티스

```python
def preprocessing_best_practices():
    """전처리 베스트 프랙티스"""
    print("=== 전처리 베스트 프랙티스 ===")
    
    # 1. 0값 처리 (로그 변환 시)
    data_with_zero = np.array([0, 1, 10, 100])
    print("1. 0값이 있는 데이터의 로그 변환:")
    print(f"   원본: {data_with_zero}")
    
    # ❌ 문제가 되는 방법
    try:
        bad_log = np.log(data_with_zero)
        print(f"   np.log() 직접: {bad_log}")  # -inf 발생!
    except:
        print("   np.log() 직접: 경고 발생!")
    
    # ✅ 올바른 방법들
    safe_log1 = np.log(data_with_zero + 1)  # +1 추가
    safe_log2 = np.log1p(data_with_zero)    # log1p 사용 (더 정확)
    print(f"   np.log(x+1): {safe_log1}")
    print(f"   np.log1p(x): {safe_log2}")
    
    # 2. 훈련/테스트 분할 후 전처리
    print("\n2. 데이터 누출 방지:")
    print("   ❌ 전체 데이터로 통계 계산 → 분할")
    print("   ✅ 분할 → 훈련 데이터로만 통계 계산 → 테스트에 적용")
    
    # 3. 역변환 가능성 고려
    original = np.array([1, 10, 100])
    normalized = (original - original.min()) / (original.max() - original.min())
    
    # 역변환
    denormalized = normalized * (original.max() - original.min()) + original.min()
    print(f"\n3. 역변환 확인:")
    print(f"   원본: {original}")
    print(f"   정규화: {normalized}")
    print(f"   역변환: {denormalized}")
    print(f"   일치 여부: {np.allclose(original, denormalized)}")

preprocessing_best_practices()
```

### 6.2 전처리 파이프라인 클래스

```python
class DataPreprocessor:
    """데이터 전처리 파이프라인 클래스"""
    
    def __init__(self):
        self.scalers = {}
        self.methods = {}
    
    def fit_transform(self, data, method='minmax', feature_names=None):
        """데이터 학습 및 변환"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        transformed_data = np.zeros_like(data, dtype=float)
        
        for i, name in enumerate(feature_names):
            col_data = data[:, i]
            
            if method == 'minmax':
                min_val, max_val = col_data.min(), col_data.max()
                self.scalers[name] = {'min': min_val, 'max': max_val}
                transformed_data[:, i] = (col_data - min_val) / (max_val - min_val)
                
            elif method == 'zscore':
                mean_val, std_val = col_data.mean(), col_data.std()
                self.scalers[name] = {'mean': mean_val, 'std': std_val}
                transformed_data[:, i] = (col_data - mean_val) / std_val
                
            elif method == 'log_zscore':
                log_data = np.log1p(col_data)  # log(1+x)
                mean_val, std_val = log_data.mean(), log_data.std()
                self.scalers[name] = {'mean': mean_val, 'std': std_val}
                transformed_data[:, i] = (log_data - mean_val) / std_val
            
            self.methods[name] = method
        
        return transformed_data
    
    def transform(self, data, feature_names=None):
        """새로운 데이터 변환 (이미 학습된 스케일러 사용)"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(data.shape[1])]
        
        transformed_data = np.zeros_like(data, dtype=float)
        
        for i, name in enumerate(feature_names):
            col_data = data[:, i]
            method = self.methods[name]
            scaler = self.scalers[name]
            
            if method == 'minmax':
                transformed_data[:, i] = (col_data - scaler['min']) / (scaler['max'] - scaler['min'])
            elif method == 'zscore':
                transformed_data[:, i] = (col_data - scaler['mean']) / scaler['std']
            elif method == 'log_zscore':
                log_data = np.log1p(col_data)
                transformed_data[:, i] = (log_data - scaler['mean']) / scaler['std']
        
        return transformed_data
```

### 6.3 전처리기 사용 예제

```python
def test_preprocessor():
    """전처리기 테스트"""
    print("=== 전처리 파이프라인 테스트 ===")
    
    # 훈련 데이터
    train_data = np.array([
        [170, 30000],
        [180, 50000], 
        [160, 25000]
    ])
    
    # 테스트 데이터
    test_data = np.array([
        [175, 40000],
        [165, 35000]
    ])
    
    # 전처리기 생성 및 훈련
    preprocessor = DataPreprocessor()
    train_scaled = preprocessor.fit_transform(train_data, method='minmax', 
                                            feature_names=['키', '연봉'])
    
    # 테스트 데이터 변환
    test_scaled = preprocessor.transform(test_data, feature_names=['키', '연봉'])
    
    print(f"훈련 데이터 (원본): \n{train_data}")
    print(f"훈련 데이터 (정규화): \n{train_scaled}")
    print(f"테스트 데이터 (원본): \n{test_data}")
    print(f"테스트 데이터 (정규화): \n{test_scaled}")

test_preprocessor()
```

---

## 7. 정리 및 요약

### 7.1 핵심 개념 정리

```python
def summary_guide():
    """핵심 개념 요약"""
    print("=== 데이터 전처리 핵심 정리 ===")
    
    print("1. 로그 변환:")
    print("   • 목적: 왜곡된 분포 개선, 스케일 차이 완화")
    print("   • 함수: np.log() (자연로그), np.log10() (상용로그)")
    print("   • 주의: 0값 처리 (np.log1p 사용)")
    
    print("\n2. 정규화 (Min-Max):")
    print("   • 목적: 데이터를 특정 범위(0~1)로 변환")
    print("   • 공식: (x - min) / (max - min)")
    print("   • 사용: 신경망, 거리 기반 알고리즘")
    
    print("\n3. 표준화 (Z-score):")
    print("   • 목적: 평균 0, 표준편차 1로 변환")
    print("   • 공식: (x - 평균) / 표준편차")
    print("   • 사용: 선형 회귀, PCA, 정규분포 가정")
    
    print("\n4. 베스트 프랙티스:")
    print("   • 훈련/테스트 분할 후 전처리")
    print("   • 0값과 이상치 처리")
    print("   • 역변환 가능성 고려")
    print("   • 적절한 방법 선택")

summary_guide()
```

### 7.2 선택 가이드

| 상황 | 추천 방법 | 이유 |
|------|-----------|------|
| 이미지 데이터 | Min-Max (0~1) | 픽셀값 범위 통일 |
| 신경망 입력 | Min-Max 또는 Z-score | 그래디언트 안정성 |
| 선형 회귀 | Z-score | 계수 해석 용이 |
| 거리 기반 알고리즘 | Min-Max | 모든 특성 동등 가중 |
| 왜곡된 분포 | 로그 + Z-score | 분포 정규화 |
| 이상치 많음 | Robust Scaling | 이상치에 강함 |

---

## 8. 완전한 실행 예제

```python
import numpy as np

# NumPy 출력 옵션 설정
np.set_printoptions(suppress=True, precision=6)

def test():
    values = np.array([3.45, 34.5, 0.01, 10, 100, 1000])
    print(np.log2(3.45), np.log10(3.45), np.log(3.45))
    
    print('원본자료:', values)
    log_values = np.log10(values)  # 상용로그
    print('로그변환 자료:', log_values)
    ln_values = np.log(values)     # 자연로그
    print('자연로그 변환 자료:', ln_values)
    print('로그변환 자료 shape:', log_values.shape)  # (6,)
    
    # 로그값의 최소 최대를 0~1 사이 범위로 정규화
    # 데이터를 일정 범위에 들어오게 하는 방법
    # 표준화: 표준편차로 요소값 마이너스 평균 (표준편차로 나눠주는것) 값을 평균을 기준으로 분포시킴
    # 정규화: 데이터의 범위를 0에서 1사이로 변환해 데이터 분포를 조정하는 방법
    min_log = log_values.min()
    max_log = log_values.max()
    normalized = (log_values - min_log) / (max_log - min_log)
    print('정규화된 로그변환 자료:', normalized)

def log_inverse():
    """로그 변환의 역변환"""
    offset = 1.0  # 로그 변환을 위한 오프셋 (0 이하 값 방지)
    log_values = np.log(10 + offset)
    original_values = np.exp(log_values) - offset  # 역변환
    print('역변환된 자료:', original_values)

class LogTrans:
    def __init__(self, offset=1.0):
        self.offset = offset
        # 로그 변환을 위한 오프셋 (0 이하 값 방지)
    
    # 로그변환 수행 메소드
    def transform(self, x: np.ndarray):
        return np.log(x + self.offset)
    
    # 역변환 메소드
    def inverse_trans(self, x_log: np.ndarray):
        return np.exp(x_log) - self.offset

def gogo():
    print('~' * 20)
    data = np.array([0.001, 1, 10, 100, 1000, 10000])
    
    # 로그 변환용 클래스 객체 생성
    log_trans = LogTrans(offset=0.001)
    
    # 데이터를 로그변환 하고 역변환을 한다
    data_log_scaled = log_trans.transform(data)
    recovered_data = log_trans.inverse_trans(data_log_scaled)
    
    print('원본 데이터:', data)
    print('로그 변환된 데이터:', data_log_scaled)
    print('역변환된 데이터:', recovered_data)

if __name__ == "__main__":
    test()
    print()
    log_inverse()
    print()
    gogo()
```

---

## 마무리

데이터 전처리는 머신러닝 성능에 직접적인 영향을 미치는 중요한 단계입니다. 각 방법의 특성을 이해하고 데이터의 특성에 맞는 적절한 전처리 방법을 선택하는 것이 중요합니다.

### 핵심 원칙:

1. **데이터를 먼저 이해하고**
2. **목적에 맞는 방법을 선택하며**
3. **항상 검증하고 역변환 가능성을 고려하세요!**

### 스케일링의 핵심:
- **로그 변환**: 편차가 큰 데이터를 처리하여 분포 개선
- **정규화**: 데이터 범위를 0~1로 축소
- **표준화**: 평균 0, 표준편차 1로 분포 조정
- **역변환**: 처리된 결과를 원본 스케일로 복원