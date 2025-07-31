import numpy as np

print("=== 차원 떨어뜨리기 vs 차원 유지하기 ===")
b = np.array([[1, 2, 3], [4, 5, 6]])

# 차원이 떨어지는 경우
print("차원이 떨어지는 인덱싱:")
print(f"b[0]: {b[0]} -> shape: {b[0].shape}")        # (3,) - 1차원
print(f"b[:, 0]: {b[:, 0]} -> shape: {b[:, 0].shape}")  # (2,) - 1차원

# 차원을 유지하는 경우
print("\n차원을 유지하는 슬라이싱:")
print(f"b[0:1]: {b[0:1]} -> shape: {b[0:1].shape}")     # (1, 3) - 2차원 유지
print(f"b[:, 0:1]: \n{b[:, 0:1]} -> shape: {b[:, 0:1].shape}")  # (2, 1) - 2차원 유지

print("\n=== 실제 데이터 분석 예제 ===")

# 학생 성적 데이터 (행: 학생, 열: 과목)
scores = np.array([
    [85, 90, 78],  # 학생1: 수학, 영어, 과학
    [92, 88, 95],  # 학생2
    [76, 82, 89],  # 학생3
    [88, 94, 91]   # 학생4
])
print(f"성적 데이터 shape: {scores.shape}")  # (4, 3)

# 1. 특정 학생의 모든 성적 (차원 떨어짐)
student1_scores = scores[0]  # 1차원 배열
print(f"학생1 성적: {student1_scores} -> shape: {student1_scores.shape}")

# 2. 특정 과목의 모든 학생 성적 (차원 떨어짐)
math_scores = scores[:, 0]  # 1차원 배열
print(f"수학 성적: {math_scores} -> shape: {math_scores.shape}")

# 3. 행별 처리 (차원 떨어뜨리기 활용)
print("\n각 학생별 평균 계산:")
for i, student_scores in enumerate(scores):  # student_scores는 1차원
    avg = np.mean(student_scores)
    print(f"학생{i+1} 평균: {avg:.1f}")

print("\n=== 머신러닝에서의 활용 ===")

# 특성 행렬 X (샘플 x 특성)
X = np.array([
    [1.2, 3.4, 2.1],  # 샘플1
    [2.1, 1.8, 3.2],  # 샘플2
    [3.0, 2.5, 1.7],  # 샘플3
])
print(f"특성 행렬 X shape: {X.shape}")  # (3, 3)

# 1. 한 샘플의 모든 특성값 추출
sample1_features = X[0]  # 차원 떨어짐: (3, 3) -> (3,)
print(f"샘플1 특성값: {sample1_features}")

# 2. 한 특성의 모든 샘플값 추출  
feature1_values = X[:, 0]  # 차원 떨어짐: (3, 3) -> (3,)
print(f"특성1 모든 값: {feature1_values}")

print("\n=== 이미지 처리 예제 ===")

# 3차원 이미지 데이터 (높이, 너비, 채널)
image = np.random.randint(0, 256, size=(4, 4, 3))  # 4x4 RGB 이미지
print(f"이미지 shape: {image.shape}")  # (4, 4, 3)

# 1. 한 픽셀의 RGB 값 (3차원 -> 1차원)
pixel_rgb = image[0, 0]  # 첫 번째 픽셀의 RGB
print(f"첫 픽셀 RGB: {pixel_rgb} -> shape: {pixel_rgb.shape}")  # (3,)

# 2. 한 행의 모든 픽셀 (3차원 -> 2차원)
row_pixels = image[0]  # 첫 번째 행
print(f"첫 행 픽셀들 shape: {row_pixels.shape}")  # (4, 3)

# 3. 한 채널의 전체 이미지 (3차원 -> 2차원)
red_channel = image[:, :, 0]  # 빨간색 채널만
print(f"빨간색 채널 shape: {red_channel.shape}")  # (4, 4)

print("\n=== 주의사항과 팁 ===")

# 1. 의도하지 않은 차원 감소 방지
arr_2d = np.array([[1, 2, 3]])
print(f"원본: {arr_2d.shape}")  # (1, 3)

# 잘못된 방법 - 차원이 떨어짐
wrong = arr_2d[0]
print(f"잘못된 방법: {wrong.shape}")  # (3,) - 1차원으로 떨어짐

# 올바른 방법 - 차원 유지
correct = arr_2d[0:1]
print(f"올바른 방법: {correct.shape}")  # (1, 3) - 2차원 유지

# 2. newaxis를 이용한 차원 확장
print("\nnewaxis를 이용한 차원 확장:")
vec = np.array([1, 2, 3])  # (3,)
col_vec = vec[:, np.newaxis]  # (3, 1) - 열벡터로 변환
row_vec = vec[np.newaxis, :]  # (1, 3) - 행벡터로 변환

print(f"원본: {vec.shape}")
print(f"열벡터: {col_vec.shape}")
print(f"행벡터: {row_vec.shape}")

print("\n=== 실전 디버깅 팁 ===")
def debug_shape(arr, name):
    """배열의 shape 정보를 출력하는 디버깅 함수"""
    print(f"{name}: shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}")

# 사용 예
data = np.array([[1, 2, 3], [4, 5, 6]])
debug_shape(data, "원본 데이터")
debug_shape(data[0], "첫 번째 행")
debug_shape(data[:, 0], "첫 번째 열")
debug_shape(data[0, 0], "첫 번째 요소")