import numpy as np

print("=== 불리언 인덱싱(Boolean Indexing) 기초 ===")

# 2차원 배열 생성
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(f"원본 배열 a:")
print(a)
print()

# 불리언 마스크 생성
bool_idx = (a >= 5)  # 5 이상인 요소들에 대해 True/False
print(f"불리언 마스크 (a >= 5):")
print(bool_idx)
print(f"마스크 shape: {bool_idx.shape}")
print(f"마스크 dtype: {bool_idx.dtype}")
print()

# 불리언 인덱싱으로 조건에 맞는 요소 선택
selected = a[bool_idx]
print(f"선택된 요소들 (5 이상): {selected}")
print(f"선택된 요소들 shape: {selected.shape}")  # 1차원 배열로 반환
print()

print("=== 불리언 인덱싱의 동작 원리 ===")
print("1. 조건식 → 불리언 마스크 생성")
print("2. 마스크에서 True인 위치의 요소만 추출")
print("3. 결과는 항상 1차원 배열")
print()

# 다양한 조건 예제
print("=== 다양한 조건 예제 ===")

data = np.array([1, -2, 3, -4, 5, -6, 7, 8, 9, 10])
print(f"데이터: {data}")

# 1. 양수만 선택
positive = data[data > 0]
print(f"양수: {positive}")

# 2. 음수만 선택
negative = data[data < 0]
print(f"음수: {negative}")

# 3. 절댓값이 5 이상
abs_large = data[np.abs(data) >= 5]
print(f"절댓값 5 이상: {abs_large}")

# 4. 짝수만 선택
even = data[data % 2 == 0]
print(f"짝수: {even}")

# 5. 특정 범위 (3 이상 8 이하)
range_data = data[(data >= 3) & (data <= 8)]
print(f"3-8 범위: {range_data}")
print()

print("=== 논리 연산자 사용 ===")
print("주의: Python의 and, or, not 대신 &, |, ~ 사용!")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"데이터: {arr}")

# 논리 AND (&)
condition1 = (arr > 3) & (arr < 8)
result1 = arr[condition1]
print(f"3 < x < 8: {result1}")

# 논리 OR (|)  
condition2 = (arr < 3) | (arr > 8)
result2 = arr[condition2]
print(f"x < 3 또는 x > 8: {result2}")

# 논리 NOT (~)
condition3 = ~(arr % 2 == 0)  # 홀수
result3 = arr[condition3]
print(f"홀수 (~짝수): {result3}")

# 복합 조건
condition4 = ((arr % 2 == 0) & (arr > 5)) | (arr == 1)
result4 = arr[condition4]
print(f"(짝수 AND >5) OR ==1: {result4}")
print()

print("=== 2차원 배열에서 불리언 인덱싱 ===")

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
print(f"행렬:")
print(matrix)

# 전체 행렬에서 조건에 맞는 요소
mask = matrix > 6
print(f"\n6보다 큰 요소들의 마스크:")
print(mask)

selected = matrix[mask]
print(f"선택된 요소들 (1차원): {selected}")

# 행 단위로 조건 적용
row_condition = matrix.sum(axis=1) > 20  # 각 행의 합이 20 초과
print(f"\n각 행 합계: {matrix.sum(axis=1)}")
print(f"합계 > 20인 행: {row_condition}")
selected_rows = matrix[row_condition]
print(f"선택된 행들:")
print(selected_rows)
print()

print("=== 실무 활용 예제 ===")

# 예제 1: 학생 성적 분석
print("1️⃣ 학생 성적 분석")
scores = np.array([
    [85, 90, 78],  # 학생1: 수학, 영어, 과학
    [92, 88, 95],  # 학생2
    [76, 82, 89],  # 학생3
    [68, 75, 71],  # 학생4
    [94, 96, 91]   # 학생5
])
student_names = ['김철수', '이영희', '박민수', '정수진', '최우수']

print("성적 데이터:")
for i, name in enumerate(student_names):
    print(f"{name}: {scores[i]}")

# 평균 80점 이상 학생
avg_scores = scores.mean(axis=1)
high_performers = avg_scores >= 80
print(f"\n각 학생 평균: {avg_scores}")
print(f"우수 학생 (평균 80+): {np.array(student_names)[high_performers]}")

# 수학 점수 90점 이상
math_excellent = scores[:, 0] >= 90
print(f"수학 우수 학생: {np.array(student_names)[math_excellent]}")

# 모든 과목 75점 이상
all_good = (scores >= 75).all(axis=1)
print(f"전과목 우수 학생: {np.array(student_names)[all_good]}")
print()

# 예제 2: 센서 데이터 이상치 탐지
print("2️⃣ 센서 데이터 이상치 탐지")
np.random.seed(42)
temperature = np.random.normal(25, 3, 100)  # 평균 25도, 표준편차 3
# 몇 개 이상치 추가
temperature[10] = 45  # 고온 이상치
temperature[50] = -10  # 저온 이상치

print(f"온도 데이터 통계:")
print(f"평균: {temperature.mean():.2f}°C")
print(f"표준편차: {temperature.std():.2f}°C")

# 이상치 탐지 (평균 ± 3*표준편차 벗어난 값)
mean_temp = temperature.mean()
std_temp = temperature.std() 
outliers_mask = (temperature < mean_temp - 3*std_temp) | (temperature > mean_temp + 3*std_temp)

outliers = temperature[outliers_mask]
outlier_indices = np.where(outliers_mask)[0]

print(f"이상치 개수: {len(outliers)}")
print(f"이상치 값: {outliers}")
print(f"이상치 위치: {outlier_indices}")

# 정상 데이터만 필터링
normal_data = temperature[~outliers_mask]
print(f"정상 데이터 개수: {len(normal_data)}")
print()

# 예제 3: 주식 데이터 분석
print("3️⃣ 주식 데이터 분석")
np.random.seed(123)
dates = np.arange('2024-01-01', '2024-01-31', dtype='datetime64[D]')
prices = 100 + np.cumsum(np.random.randn(30) * 2)  # 누적합으로 가격 생성
volumes = np.random.randint(1000, 10000, 30)

print("주식 데이터 (처음 10일):")
for i in range(10):
    print(f"{dates[i]}: 가격 {prices[i]:.2f}, 거래량 {volumes[i]}")

# 고가격 고거래량 날짜 찾기
high_price = prices > prices.mean()
high_volume = volumes > volumes.mean()
target_days = high_price & high_volume

print(f"\n평균 가격: {prices.mean():.2f}")
print(f"평균 거래량: {volumes.mean():.0f}")
print(f"고가격 고거래량 날짜:")
for date in dates[target_days]:
    print(f"  {date}")
print()

print("=== 불리언 인덱싱 고급 활용 ===")

# where 함수 활용
print("1️⃣ np.where 함수")
data = np.array([1, 2, 3, 4, 5, 6])
# 조건에 따라 다른 값 할당
result = np.where(data > 3, data * 2, data)  # 3 초과면 2배, 아니면 그대로
print(f"원본: {data}")
print(f"조건적 변환: {result}")

# 다중 조건 where
result2 = np.where(data < 3, '작음', 
                  np.where(data > 4, '큼', '보통'))
print(f"다중 조건 분류: {result2}")
print()

# 조건부 집계
print("2️⃣ 조건부 집계")
sales_data = np.array([
    [100, 150, 200],  # 1월: 제품A, B, C
    [120, 180, 220],  # 2월
    [90, 140, 190],   # 3월
    [110, 160, 210]   # 4월
])

# 150 이상 매출만 합계
high_sales_mask = sales_data >= 150
high_sales_total = sales_data[high_sales_mask].sum()
print(f"매출 데이터:")
print(sales_data)
print(f"150 이상 매출 총합: {high_sales_total}")

# 제품별 목표 달성 여부 (목표: 각각 400, 600, 800)
targets = np.array([400, 600, 800])
monthly_totals = sales_data.sum(axis=0)
achieved = monthly_totals >= targets
print(f"월별 총매출: {monthly_totals}")
print(f"목표: {targets}")
print(f"목표 달성 제품: {achieved}")
print()

print("=== 성능 및 메모리 고려사항 ===")

# 큰 배열에서의 불리언 인덱싱
large_array = np.random.randn(1000000)

import time

# 방법 1: 불리언 인덱싱
start = time.time()
positive_bool = large_array[large_array > 0]
bool_time = time.time() - start

# 방법 2: where 사용
start = time.time()
positive_where = large_array[np.where(large_array > 0)]
where_time = time.time() - start

print(f"100만 개 데이터에서 양수 선택:")
print(f"불리언 인덱싱: {bool_time:.4f}초")
print(f"where 사용: {where_time:.4f}초")
print(f"선택된 요소 개수: {len(positive_bool)}")
print()

print("=== 주의사항과 팁 ===")
print()
print("⚠️ 주의사항:")
print("1. 논리 연산자: &, |, ~ 사용 (and, or, not 아님)")
print("2. 괄호 사용: (조건1) & (조건2) 형태로 우선순위 명확히")
print("3. 결과는 항상 1차원 배열")
print("4. 원본 배열과 같은 shape의 불리언 마스크 필요")
print()

print("💡 성능 팁:")
print("1. 복잡한 조건은 미리 마스크로 저장")
print("2. 여러 번 사용할 조건은 변수로 저장")
print("3. 가능하면 벡터화된 연산 사용")
print("4. 큰 데이터에서는 메모리 사용량 주의")
print()

# 잘못된 예와 올바른 예
print("❌ 잘못된 예:")
try:
    # wrong = data[data > 3 and data < 8]  # 에러 발생
    pass
except:
    print("data > 3 and data < 8  # TypeError!")

print("✅ 올바른 예:")
print("(data > 3) & (data < 8)  # 정상 동작")
print()

print("=== 실전 디버깅 팁 ===")

def debug_boolean_mask(data, mask, description=""):
    """불리언 마스크 디버깅 함수"""
    print(f"디버깅: {description}")
    print(f"데이터 shape: {data.shape}")
    print(f"마스크 shape: {mask.shape}")
    print(f"True 개수: {mask.sum()}")
    print(f"선택율: {mask.sum()/mask.size*100:.1f}%")
    print(f"선택된 값: {data[mask]}")
    print()

# 사용 예
test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_mask = (test_data > 5) & (test_data % 2 == 0)
debug_boolean_mask(test_data, test_mask, "5보다 크고 짝수인 조건")

print("=== 정리 ===")
print("🎯 불리언 인덱싱의 핵심:")
print("1. 조건식으로 True/False 마스크 생성")
print("2. 마스크를 인덱스로 사용하여 조건에 맞는 요소만 선택")
print("3. 결과는 항상 1차원 배열")
print("4. 데이터 필터링, 이상치 탐지, 조건부 분석에 핵심적")
print()
print("🚀 실무에서 매일 사용하는 필수 기능!")
print("- 데이터 전처리: 결측치, 이상치 제거")
print("- 데이터 분석: 조건부 통계, 그룹별 분석") 
print("- 머신러닝: 특성 선택, 데이터 분할")