# NumPy 인덱싱이 중요한 이유

## 왜 배워야 하는가? 🤔

NumPy 배열 인덱싱은 **데이터 사이언스의 기본기**입니다. 실제로는 이런 상황에서 매일 사용합니다:

### 1. 데이터 분석에서 특정 부분 추출
```python
# 실제 상황: 학생 성적 데이터
scores = np.array([
    [85, 90, 78],  # 학생1: 수학, 영어, 과학
    [92, 88, 95],  # 학생2
    [76, 82, 89],  # 학생3
])

# 필요한 데이터만 뽑아내기
first_student = scores[0]           # 첫 번째 학생 모든 성적
math_scores = scores[:, 0]          # 모든 학생의 수학 성적
top_students = scores[0:2]          # 상위 2명 학생 데이터
```

### 2. 이미지 처리
```python
# 실제 상황: 이미지 데이터 (높이, 너비, RGB)
image = np.random.randint(0, 255, (480, 640, 3))  # 480x640 컬러 이미지

# 이미지 조작
cropped = image[100:400, 50:350]    # 이미지 자르기
red_channel = image[:, :, 0]        # 빨간색 채널만 추출
center_pixel = image[240, 320]      # 중앙 픽셀 RGB 값
```

### 3. 머신러닝 데이터 전처리
```python
# 실제 상황: 특성 행렬과 타겟 벡터
data = np.array([
    [1.2, 3.4, 2.1, 0],  # 마지막이 레이블
    [2.1, 1.8, 3.2, 1],
    [3.0, 2.5, 1.7, 0]
])

# 특성과 레이블 분리
X = data[:, :-1]        # 모든 행, 마지막 열 제외 (특성)
y = data[:, -1]         # 모든 행, 마지막 열만 (레이블)

# 훈련/테스트 분할
train_X = X[:80]        # 처음 80% 훈련용
test_X = X[80:]         # 나머지 20% 테스트용
```

## 인덱싱 패턴 완전 분석

### 1차원 배열 인덱싱
```python
a = np.array([1, 2, 3, 4, 5])

# 기본 인덱싱
a[0]        # → 1 (첫 번째 요소)
a[-1]       # → 5 (마지막 요소)

# 슬라이싱 [시작:끝:간격]
a[1:4]      # → [2, 3, 4] (인덱스 1~3)
a[1:4:2]    # → [2, 4] (인덱스 1~3에서 2칸씩)
a[:3]       # → [1, 2, 3] (처음부터 인덱스 2까지)
a[2:]       # → [3, 4, 5] (인덱스 2부터 끝까지)
a[::2]      # → [1, 3, 5] (전체에서 2칸씩)
```

### 2차원 배열 인덱싱
```python
a = np.array([[1, 2, 3], 
              [4, 5, 6]])

# 개별 요소 접근
a[0, 0]     # → 1 (첫 번째 행, 첫 번째 열)
a[1, 2]     # → 6 (두 번째 행, 세 번째 열)

# 행 전체 선택
a[0]        # → [1, 2, 3] (첫 번째 행, 1차원)
a[0:1]      # → [[1, 2, 3]] (첫 번째 행, 2차원 유지)

# 열 전체 선택
a[:, 0]     # → [1, 4] (첫 번째 열, 1차원)
a[:, 0:1]   # → [[1], [4]] (첫 번째 열, 2차원 유지)

# 부분 영역 선택
a[1:, 0:2]  # → [[4, 5]] (두 번째 행부터, 첫~두 번째 열)
```

## 실제 활용 예제

### 예제 1: CSV 데이터 처리
```python
# 실제 상황: 매출 데이터 (월별 x 제품별)
sales_data = np.array([
    [100, 120, 90],   # 1월: 제품A, B, C
    [110, 130, 95],   # 2월
    [105, 125, 100],  # 3월
    [115, 140, 110]   # 4월
])

# 분석 작업
q1_sales = sales_data[0:3]          # 1분기 데이터
product_a = sales_data[:, 0]        # 제품A 월별 매출
march_data = sales_data[2]          # 3월 전체 데이터
best_month = sales_data[np.argmax(sales_data.sum(axis=1))]  # 최고 매출 월
```

### 예제 2: 센서 데이터 분석
```python
# 실제 상황: 시간별 온도, 습도, 압력 센서 데이터
sensor_data = np.random.normal([25, 60, 1013], [2, 5, 10], (24, 3))  # 24시간 데이터

# 분석 작업
temperature = sensor_data[:, 0]     # 온도 데이터만
morning_data = sensor_data[6:12]    # 오전 6시~11시 데이터
afternoon_temp = sensor_data[12:18, 0]  # 오후 온도만
night_conditions = sensor_data[22:]  # 밤 10시 이후 모든 센서
```

### 예제 3: 이미지 필터링
```python
# 실제 상황: 이미지에서 관심 영역(ROI) 추출
image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 RGB 이미지

# 이미지 처리 작업
roi = image[20:80, 30:70]           # 관심 영역 추출
grayscale = image[:, :, 0]          # 첫 번째 채널만 (흑백)
top_half = image[:50, :]            # 이미지 상단 절반
border = image[[0, -1], :]          # 첫 번째와 마지막 행 (테두리)
```

## 왜 이것이 중요한가?

### 1. **메모리 효율성**
```python
# 나쁜 예: 새로운 배열 생성
filtered_data = []
for i in range(len(large_array)):
    if condition(large_array[i]):
        filtered_data.append(large_array[i])

# 좋은 예: 인덱싱으로 직접 접근
mask = large_array > threshold
filtered_data = large_array[mask]  # 훨씬 빠르고 메모리 효율적
```

### 2. **코드 간결성**
```python
# 복잡한 반복문 대신
result = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if i > j:
            result.append(data[i, j])

# 간단한 인덱싱으로
result = data[np.triu_indices_from(data, k=1)]
```

### 3. **성능 향상**
```python
import time

# 반복문 방식 (느림)
start = time.time()
result1 = []
for i in range(10000):
    result1.append(data[i] * 2)
loop_time = time.time() - start

# 벡터화 방식 (빠름)
start = time.time()
result2 = data * 2
vector_time = time.time() - start

print(f"반복문: {loop_time:.4f}초")
print(f"벡터화: {vector_time:.4f}초")
print(f"속도 향상: {loop_time/vector_time:.1f}배")
```

## 고급 인덱싱 패턴

### 불린 인덱싱 (매우 중요!)
```python
data = np.array([1, 2, -3, 4, -5, 6])

# 조건에 맞는 데이터만 선택
positive = data[data > 0]           # → [1, 2, 4, 6]
even_numbers = data[data % 2 == 0]  # → [2, 4, 6]

# 실제 활용: 이상치 제거
temperatures = np.array([20, 22, 19, 150, 21, 23, -50, 24])
normal_temps = temperatures[(temperatures > -10) & (temperatures < 50)]
```

### 팬시 인덱싱
```python
data = np.array([10, 20, 30, 40, 50])
indices = [0, 2, 4]
selected = data[indices]  # → [10, 30, 50]

# 실제 활용: 특정 샘플만 선택
sample_indices = np.random.choice(len(dataset), 100, replace=False)
sample_data = dataset[sample_indices]
```

## 실무에서 자주 쓰는 패턴들

### 데이터 분할
```python
# 시계열 데이터를 훈련/검증/테스트로 분할
total_length = len(time_series)
train_end = int(0.7 * total_length)
val_end = int(0.9 * total_length)

train_data = time_series[:train_end]
val_data = time_series[train_end:val_end]
test_data = time_series[val_end:]
```

### 배치 처리
```python
# 대용량 데이터를 배치 단위로 처리
batch_size = 32
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    process_batch(batch)
```

### 윈도우 슬라이딩
```python
# 시계열 데이터에서 슬라이딩 윈도우 생성
window_size = 5
windows = []
for i in range(len(data) - window_size + 1):
    window = data[i:i+window_size]
    windows.append(window)
```

## 결론

NumPy 인덱싱을 배우는 이유:

1. **실무 필수 기능**: 데이터 분석의 90% 이상에서 사용
2. **성능 최적화**: 반복문 없이 빠른 데이터 처리
3. **코드 간결성**: 복잡한 로직을 한 줄로 표현
4. **메모리 효율성**: 불필요한 복사 없이 데이터 접근
5. **라이브러리 호환성**: Pandas, Sklearn 등 모든 라이브러리의 기초
