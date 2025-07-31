import numpy as np

print("=== 복사(Copy) 동작 확인 ===")
aa = np.array([1, 2, 3, 4, 5])
cc = aa[1:3].copy()  # 복사본 생성

print(f"원본 aa: {aa}")
print(f"복사본 cc: {cc}")

cc[0] = 55  # 복사본 수정
print(f"\ncc[0] = 55 수정 후:")
print(f"cc: {cc}")           # 복사본만 변경
print(f"aa: {aa}")           # 원본은 변경되지 않음
print(f"aa[1:3]: {aa[1:3]}")  # 원본의 해당 부분 확인
print()

print("=== 핵심! 차원 유지 vs 차원 감소 ===")

# 2차원 배열 생성
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(f"원본 2차원 배열 a:")
print(a)
print(f"a.shape: {a.shape}")  # (2, 3)
print()

print("🔍 행(Row) 선택 비교:")
print("=" * 50)

# 행 선택 - 차원 감소 vs 유지
r1 = a[1, :]     # 인덱싱: 차원 감소 (2D → 1D)
r2 = a[1:2, :]   # 슬라이싱: 차원 유지 (2D → 2D)

print(f"r1 = a[1, :] (인덱싱):")
print(f"  값: {r1}")
print(f"  shape: {r1.shape}")  # (3,) - 1차원
print(f"  차원수: {r1.ndim}차원")

print(f"\nr2 = a[1:2, :] (슬라이싱):")
print(f"  값: {r2}")
print(f"  shape: {r2.shape}")  # (1, 3) - 2차원 유지
print(f"  차원수: {r2.ndim}차원")
print()

print("🔍 열(Column) 선택 비교:")
print("=" * 50)

# 열 선택 - 차원 감소 vs 유지
c1 = a[:, 1]     # 인덱싱: 차원 감소 (2D → 1D)
c2 = a[:, 1:2]   # 슬라이싱: 차원 유지 (2D → 2D)

print(f"c1 = a[:, 1] (인덱싱):")
print(f"  값: {c1}")
print(f"  shape: {c1.shape}")  # (2,) - 1차원
print(f"  차원수: {c1.ndim}차원")

print(f"\nc2 = a[:, 1:2] (슬라이싱):")
print(f"  값: {c2}")
print(f"  shape: {c2.shape}")  # (2, 1) - 2차원 유지
print(f"  차원수: {c2.ndim}차원")
print()

print("📋 규칙 정리:")
print("=" * 50)
print("✅ 인덱싱 (단일 숫자): 차원 감소")
print("   a[1, :] → (3,)     # 2D → 1D")
print("   a[:, 1] → (2,)     # 2D → 1D")
print()
print("✅ 슬라이싱 (범위): 차원 유지")
print("   a[1:2, :] → (1, 3) # 2D → 2D")
print("   a[:, 1:2] → (2, 1) # 2D → 2D")
print()

print("=== 실제 활용 예제 ===")

# 학생 성적 데이터
scores = np.array([
    [85, 90, 78],  # 학생1: 수학, 영어, 과학
    [92, 88, 95],  # 학생2
    [76, 82, 89],  # 학생3
])
print(f"학생 성적 데이터:")
print(scores)
print()

print("1️⃣ 특정 학생 성적 추출:")
student1_1d = scores[0, :]     # 1차원 벡터
student1_2d = scores[0:1, :]   # 2차원 행렬

print(f"학생1 성적 (1D): {student1_1d}, shape: {student1_1d.shape}")
print(f"학생1 성적 (2D): {student1_2d}, shape: {student1_2d.shape}")
print()

print("2️⃣ 특정 과목 성적 추출:")
math_1d = scores[:, 0]       # 1차원 벡터
math_2d = scores[:, 0:1]     # 2차원 행렬

print(f"수학 성적 (1D): {math_1d}, shape: {math_1d.shape}")
print(f"수학 성적 (2D): {math_2d}, shape: {math_2d.shape}")
print()

print("=== 언제 어떤 방법을 사용할까? ===")
print()

print("🎯 1차원이 필요한 경우:")
print("- 벡터 연산 (내적, 코사인 유사도)")
print("- 1차원 함수 입력 (평균, 표준편차)")
print("- 그래프 플롯팅")
print("- 반복문에서 개별 요소 처리")

# 예제: 벡터 연산
print(f"\n예제 - 벡터 내적:")
vec1 = scores[0, :]  # 학생1 성적 (1D)
vec2 = scores[1, :]  # 학생2 성적 (1D)
dot_product = np.dot(vec1, vec2)
print(f"학생1 성적: {vec1}")
print(f"학생2 성적: {vec2}")
print(f"내적 결과: {dot_product}")
print()

print("🎯 2차원이 필요한 경우:")
print("- 행렬 연산 (곱셈, 전치)")
print("- 다른 2차원 배열과 연산")
print("- 브로드캐스팅")
print("- 데이터프레임과 호환")

# 예제: 행렬 연산
print(f"\n예제 - 브로드캐스팅:")
weights = np.array([[0.3], [0.4], [0.3]])  # 가중치 (3x1)
student1_matrix = scores[0:1, :]           # 학생1 (1x3)

print(f"학생1 성적 (2D): {student1_matrix}")
print(f"가중치: {weights.flatten()}")
weighted_score = student1_matrix @ weights  # 행렬 곱셈
print(f"가중 평균: {weighted_score[0, 0]:.2f}")
print()

print("=== 3차원 배열 예제 ===")

# 3차원 배열 (이미지 배치 예제)
images = np.random.randint(0, 256, (3, 4, 4))  # 3장의 4x4 이미지
print(f"이미지 배치 shape: {images.shape}")  # (3, 4, 4)
print()

print("🖼️ 이미지 선택 방법:")
# 첫 번째 이미지 선택
img1_2d = images[0]        # 차원 감소: (3,4,4) → (4,4)
img1_3d = images[0:1]      # 차원 유지: (3,4,4) → (1,4,4)

print(f"첫 번째 이미지 (2D): shape={img1_2d.shape}")
print(f"첫 번째 이미지 (3D): shape={img1_3d.shape}")
print()

print("=== 차원 변환 팁 ===")

# newaxis 사용한 차원 확장
vector = np.array([1, 2, 3])
print(f"원본 벡터: {vector}, shape: {vector.shape}")

# 행 벡터로 변환
row_vector = vector[np.newaxis, :]
print(f"행 벡터: {row_vector}, shape: {row_vector.shape}")

# 열 벡터로 변환
col_vector = vector[:, np.newaxis]
print(f"열 벡터:\n{col_vector}\nshape: {col_vector.shape}")
print()

# reshape 사용
reshaped = vector.reshape(1, -1)  # 행 벡터
print(f"reshape로 행 벡터: {reshaped}, shape: {reshaped.shape}")

reshaped = vector.reshape(-1, 1)  # 열 벡터
print(f"reshape로 열 벡터:\n{reshaped}\nshape: {reshaped.shape}")
print()

print("=== 실무에서 자주 만나는 상황 ===")

# 머신러닝 데이터 처리
print("1️⃣ 머신러닝 특성 추출:")
data = np.array([
    [1.0, 2.0, 3.0, 0],  # 특성1, 특성2, 특성3, 레이블
    [1.5, 2.5, 3.5, 1],
    [2.0, 3.0, 4.0, 0]
])

# 잘못된 방법 - 차원 문제 발생 가능
features_wrong = data[:, :-1]  # 올바름 (2D 유지)
single_feature_wrong = data[:, 0]  # 1D 벡터

print(f"전체 특성: {features_wrong.shape}")
print(f"첫 번째 특성 (1D): {single_feature_wrong.shape}")

# 올바른 방법 - 차원 일관성 유지
single_feature_right = data[:, 0:1]  # 2D 유지
print(f"첫 번째 특성 (2D): {single_feature_right.shape}")
print()

print("2️⃣ 시계열 데이터 처리:")
time_series = np.random.randn(100, 5)  # 100 시점, 5개 변수

# 특정 시점 데이터
point_1d = time_series[50, :]     # (5,) - 1D
point_2d = time_series[50:51, :]  # (1, 5) - 2D

print(f"특정 시점 (1D): {point_1d.shape}")
print(f"특정 시점 (2D): {point_2d.shape}")

# 모델 입력으로 사용 시
print("모델 입력 시 고려사항:")
print("- 많은 ML 모델이 2D 입력 기대 (samples, features)")
print("- 1D 벡터는 reshape 또는 슬라이싱으로 2D 변환 필요")
print()

print("=== 디버깅 팁 ===")

def debug_array_info(arr, name):
    """배열 정보 출력 함수"""
    print(f"{name}:")
    print(f"  값: {arr}")
    print(f"  shape: {arr.shape}")
    print(f"  차원: {arr.ndim}D")
    print(f"  타입: {arr.dtype}")
    print()

# 사용 예제
test_array = np.array([[1, 2], [3, 4]])
debug_array_info(test_array, "원본 배열")
debug_array_info(test_array[0], "첫 번째 행 (인덱싱)")
debug_array_info(test_array[0:1], "첫 번째 행 (슬라이싱)")

print("=== 정리 ===")
print("🔑 핵심 규칙:")
print("1. 인덱싱 (단일 숫자) → 차원 감소")
print("2. 슬라이싱 (범위) → 차원 유지")
print("3. 목적에 따라 선택:")
print("   - 벡터 연산 → 1D 사용")
print("   - 행렬 연산, 일관성 → 2D 사용")
print("4. 항상 .shape 확인하는 습관!")
print()
print("💡 실무 팁:")
print("- 차원 일관성이 중요한 프로젝트에서는 슬라이싱 선호")
print("- 성능이 중요한 계산에서는 인덱싱으로 차원 축소")
print("- 디버깅 시 반드시 shape 확인!")