import numpy as np

print("=== 2차원 배열 생성 ===")
# 2차원 배열 생성
b = np.array([[1, 2, 3], 
              [4, 5, 6]])

print(f"원본 배열 b:")
print(b)
print(f"b.shape: {b.shape}")  # (2, 3) - 2행 3열
print(f"b.ndim: {b.ndim}")    # 2차원

print("\n=== 차원 떨어뜨리기 (인덱싱) ===")
# 첫 번째 행을 선택 -> 2차원에서 1차원으로 떨어짐
print(f"b[0]: {b[0]}")           # [1 2 3]
print(f"b[0].shape: {b[0].shape}")  # (3,) - 1차원 배열
print(f"b[0].ndim: {b[0].ndim}")    # 1차원

print(f"b[1]: {b[1]}")           # [4 5 6]
print(f"b[1].shape: {b[1].shape}")  # (3,) - 1차원 배열

print("\n=== 개별 요소 접근 ===")
print(f"b[0][0]: {b[0][0]}")     # 1 (스칼라, 0차원)
print(f"b[0][1]: {b[0][1]}")     # 2
print(f"b[1][2]: {b[1][2]}")     # 6

# 또는 다음과 같이 접근 가능
print(f"b[0, 0]: {b[0, 0]}")     # 1
print(f"b[0, 1]: {b[0, 1]}")     # 2
print(f"b[1, 2]: {b[1, 2]}")     # 6

print("\n=== 다양한 차원 떨어뜨리기 방법 ===")

# 1. 슬라이싱으로 행 선택
print("행 슬라이싱:")
print(f"b[0:1]: \n{b[0:1]}")         # 2차원 유지 [[1 2 3]]
print(f"b[0:1].shape: {b[0:1].shape}")  # (1, 3)

print(f"b[0]: {b[0]}")               # 1차원 [1 2 3]
print(f"b[0].shape: {b[0].shape}")      # (3,)

# 2. 열 선택
print("\n열 선택:")
print(f"b[:, 0]: {b[:, 0]}")         # [1 4] - 첫 번째 열
print(f"b[:, 0].shape: {b[:, 0].shape}")  # (2,) - 1차원

print(f"b[:, 1]: {b[:, 1]}")         # [2 5] - 두 번째 열
print(f"b[:, 1].shape: {b[:, 1].shape}")  # (2,) - 1차원

print("\n=== 3차원 배열 예제 ===")
# 3차원 배열 생성
c = np.array([[[1, 2], [3, 4]], 
              [[5, 6], [7, 8]]])

print(f"3차원 배열 c:")
print(c)
print(f"c.shape: {c.shape}")    # (2, 2, 2)
print(f"c.ndim: {c.ndim}")      # 3차원

# 차원 떨어뜨리기
print(f"\nc[0]: \n{c[0]}")           # 2차원 배열
print(f"c[0].shape: {c[0].shape}")      # (2, 2)
print(f"c[0].ndim: {c[0].ndim}")        # 2차원

print(f"\nc[0][0]: {c[0][0]}")       # 1차원 배열
print(f"c[0][0].shape: {c[0][0].shape}")  # (2,)
print(f"c[0][0].ndim: {c[0][0].ndim}")    # 1차원

print(f"\nc[0][0][0]: {c[0][0][0]}")  # 스칼라 (0차원)
print(f"type(c[0][0][0]): {type(c[0][0][0])}")  # numpy.int64

print("\n=== 차원 변화 정리 ===")
print("3차원 -> 2차원 -> 1차원 -> 0차원(스칼라)")
print(f"c.shape: {c.shape} -> c[0].shape: {c[0].shape} -> c[0][0].shape: {c[0][0].shape} -> c[0][0][0]: {c[0][0][0]} (스칼라)")

print("\n=== keepdims로 차원 유지하기 ===")
# numpy 함수에서 keepdims=True 옵션 사용
d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"원본 d.shape: {d.shape}")

# 평균 계산 시 차원 변화
print(f"np.mean(d, axis=0): {np.mean(d, axis=0)}")  # [2.5 3.5 4.5]
print(f"np.mean(d, axis=0).shape: {np.mean(d, axis=0).shape}")  # (3,)

print(f"np.mean(d, axis=0, keepdims=True): {np.mean(d, axis=0, keepdims=True)}")  # [[2.5 3.5 4.5]]
print(f"np.mean(d, axis=0, keepdims=True).shape: {np.mean(d, axis=0, keepdims=True).shape}")  # (1, 3)