import numpy as np

print("=== 서브배열(Sub-array) 개념 ===")
print("서브배열 = 원본 배열의 일부분을 잘라낸 배열")
print()

# 원본 배열
aa = np.array([1, 2, 3, 4, 5])
print(f"원본 배열 aa: {aa}")
print(f"aa의 메모리 주소: {id(aa)}")
print()

# 서브배열 생성 (슬라이싱)
bb = aa[1:3]  # 인덱스 1부터 2까지 (3은 제외)
print(f"서브배열 bb = aa[1:3]: {bb}")
print(f"bb의 메모리 주소: {id(bb)}")
print(f"bb[0]: {bb[0]} (서브배열의 첫 번째 요소)")
print()

print("=== 중요! 뷰(View) vs 복사(Copy) ===")

# 1. 뷰(View) - 메모리 공유
print("1. 뷰(View) - 슬라이싱은 기본적으로 뷰를 생성")
view_array = aa[1:4]
print(f"뷰 배열: {view_array}")
print(f"원본과 메모리 공유: {view_array.base is aa}")

# 뷰를 수정하면 원본도 변경됨!
print("\n뷰 수정 테스트:")
print(f"수정 전 - 원본: {aa}, 뷰: {view_array}")
view_array[0] = 999
print(f"뷰[0] = 999 수정 후")
print(f"수정 후 - 원본: {aa}, 뷰: {view_array}")
print("→ 원본도 함께 변경됨! (메모리 공유)")

# 원본 복구
aa = np.array([1, 2, 3, 4, 5])
print()

# 2. 복사(Copy) - 독립적인 메모리
print("2. 복사(Copy) - .copy()로 독립적인 배열 생성")
copy_array = aa[1:4].copy()
print(f"복사 배열: {copy_array}")
print(f"원본과 메모리 공유: {copy_array.base is aa}")

print("\n복사 수정 테스트:")
print(f"수정 전 - 원본: {aa}, 복사: {copy_array}")
copy_array[0] = 888
print(f"복사[0] = 888 수정 후")
print(f"수정 후 - 원본: {aa}, 복사: {copy_array}")
print("→ 원본은 변경되지 않음! (독립적인 메모리)")
print()

print("=== 실제 활용 예제 ===")

# 예제 1: 이미지 처리
print("1. 이미지 ROI (Region of Interest) 처리")
image = np.random.randint(0, 256, (5, 5))  # 5x5 가상 이미지
print(f"원본 이미지:\n{image}")

# ROI 선택 (뷰)
roi = image[1:4, 1:4]  # 중앙 3x3 영역
print(f"\nROI (뷰):\n{roi}")

# ROI 밝기 조절 (원본도 함께 변경됨)
roi_bright = roi + 50
image[1:4, 1:4] = roi_bright
print(f"\n밝기 조절 후 원본 이미지:\n{image}")
print()

# 예제 2: 데이터 분석
print("2. 데이터 분석 - 특정 기간 데이터")
sales_data = np.array([100, 120, 90, 110, 130, 95, 105, 125])
print(f"전체 매출 데이터: {sales_data}")

# Q1 데이터 (1-3월)
q1_data = sales_data[0:3]  # 뷰
print(f"Q1 데이터 (뷰): {q1_data}")

# Q1 데이터를 독립적으로 분석하고 싶다면
q1_data_copy = sales_data[0:3].copy()  # 복사
print(f"Q1 데이터 (복사): {q1_data_copy}")

# 정규화 (원본에 영향 주지 않음)
q1_normalized = (q1_data_copy - q1_data_copy.mean()) / q1_data_copy.std()
print(f"Q1 정규화 데이터: {q1_normalized}")
print(f"원본 데이터 (변경 없음): {sales_data}")
print()

print("=== 뷰 vs 복사 판별 방법 ===")

arr = np.array([1, 2, 3, 4, 5])

# 뷰인지 복사인지 확인
subarray1 = arr[1:4]        # 뷰
subarray2 = arr[1:4].copy() # 복사

print(f"원본 배열: {arr}")
print(f"서브배열1 (뷰): {subarray1}")
print(f"서브배열2 (복사): {subarray2}")

print(f"\n서브배열1이 뷰인가? {subarray1.base is arr}")
print(f"서브배열2가 뷰인가? {subarray2.base is arr}")

print(f"\n서브배열1 소유 여부: {subarray1.flags.owndata}")
print(f"서브배열2 소유 여부: {subarray2.flags.owndata}")
print()

print("=== 언제 뷰를 사용하고 언제 복사를 사용할까? ===")
print()

print("🔗 뷰(View) 사용 상황:")
print("1. 메모리 절약이 중요할 때")
print("2. 원본 데이터와 연동하여 수정하고 싶을 때")
print("3. 대용량 데이터의 일부분만 참조할 때")
print("4. 읽기 전용으로 사용할 때")
print()

print("📋 복사(Copy) 사용 상황:")
print("1. 원본 데이터를 보존하고 싶을 때")
print("2. 독립적인 데이터 처리가 필요할 때")
print("3. 병렬 처리에서 안전한 데이터 분리가 필요할 때")
print("4. 임시 계산용 데이터가 필요할 때")
print()

print("=== 실무 활용 패턴 ===")

# 패턴 1: 배치 처리
print("1. 배치 처리 예제")
data = np.random.randn(1000, 10)  # 1000개 샘플, 10개 특성
batch_size = 100

for i in range(0, len(data), batch_size):
    # 뷰 사용 (메모리 효율적)
    batch = data[i:i+batch_size]
    print(f"배치 {i//batch_size + 1}: shape={batch.shape}, 뷰={batch.base is data}")

print()

# 패턴 2: 데이터 전처리
print("2. 데이터 전처리 예제")
raw_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(f"원본 데이터:\n{raw_data}")

# 특성과 타겟 분리 (뷰 사용)
features = raw_data[:, :-1]  # 마지막 열 제외
targets = raw_data[:, -1]    # 마지막 열만

print(f"특성 (뷰):\n{features}")
print(f"타겟 (뷰): {targets}")

# 전처리용 복사본 생성
features_processed = features.copy()
features_processed = features_processed / features_processed.max()  # 정규화

print(f"전처리된 특성:\n{features_processed}")
print(f"원본 특성 (변경 없음):\n{features}")
print()

# 패턴 3: 시계열 윈도우
print("3. 시계열 윈도우 예제")
time_series = np.array([1, 4, 7, 2, 8, 3, 9, 5, 6])
window_size = 3

print(f"시계열 데이터: {time_series}")
print("슬라이딩 윈도우:")

for i in range(len(time_series) - window_size + 1):
    window = time_series[i:i+window_size]  # 뷰
    print(f"윈도우 {i}: {window} (뷰={window.base is time_series})")

print()

print("=== 주의사항과 팁 ===")
print()

print("⚠️ 주의사항:")
print("1. 뷰 수정 시 원본도 함께 변경됨")
print("2. 원본 배열이 삭제되면 뷰도 무효화될 수 있음")
print("3. 복잡한 인덱싱은 뷰가 아닌 복사본을 만들 수 있음")
print()

print("💡 성능 팁:")
print("1. 가능하면 뷰 사용 (메모리 절약)")
print("2. 수정이 필요한 경우만 복사 사용")
print("3. 큰 배열에서는 뷰/복사 구분이 매우 중요")

# 성능 비교 예제
print("\n성능 비교:")
large_array = np.random.randn(1000000)

import time

# 뷰 생성 시간
start = time.time()
view_sub = large_array[100000:200000]
view_time = time.time() - start

# 복사 생성 시간
start = time.time()
copy_sub = large_array[100000:200000].copy()
copy_time = time.time() - start

print(f"뷰 생성 시간: {view_time:.6f}초")
print(f"복사 생성 시간: {copy_time:.6f}초")
print(f"복사가 뷰보다 {copy_time/view_time:.1f}배 느림")

print("\n=== 정리 ===")
print("서브배열 = 원본 배열의 일부분")
print("뷰 = 메모리 공유, 빠름, 원본과 연동")
print("복사 = 독립적 메모리, 느림, 원본과 분리")
print("실무에서는 목적에 따라 선택적 사용!")