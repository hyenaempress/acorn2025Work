import numpy as np
import matplotlib.pyplot as plt

print("=== 머신러닝에서 내적(Dot Product)의 핵심 역할 ===")
print()

print("🎯 내적이란?")
print("두 벡터의 대응하는 성분끼리 곱한 후 모두 더한 값")
print("공식: a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ")
print("기하학적 의미: |a||b|cos(θ) (벡터 크기 × 코사인 각도)")
print()

# 기본 내적 예제
print("=== 1. 기본 내적 계산 ===")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)
print(f"벡터 a: {a}")
print(f"벡터 b: {b}")
print(f"내적 a·b = 1×4 + 2×5 + 3×6 = {dot_product}")
print()

print("=== 2. 선형회귀: 예측의 핵심 ===")
print()

# 선형회귀 예제
print("🏠 주택 가격 예측 예제:")
print("특성: [면적, 방개수, 층수]")
print("가중치: 각 특성의 중요도")
print()

# 주택 데이터 (3개 특성)
house_features = np.array([
    [100, 3, 2],  # 집1: 100㎡, 3개방, 2층
    [80, 2, 1],   # 집2: 80㎡, 2개방, 1층  
    [120, 4, 3],  # 집3: 120㎡, 4개방, 3층
    [90, 3, 2]    # 집4: 90㎡, 3개방, 2층
])

# 학습된 가중치 (각 특성의 중요도)
weights = np.array([2.5, 30, 10])  # [면적당 250만원, 방당 3000만원, 층당 1000만원]
bias = 50  # 기본 가격 5000만원

print("주택 데이터:")
for i, house in enumerate(house_features):
    print(f"집{i+1}: 면적={house[0]}㎡, 방={house[1]}개, 층={house[2]}층")

print(f"\n학습된 가중치: {weights} (면적당 250만원, 방당 3000만원, 층당 1000만원)")
print(f"편향(bias): {bias} (기본 5000만원)")
print()

# 예측 계산 (내적 사용!)
predictions = np.dot(house_features, weights) + bias
print("예측 과정:")
for i, (house, pred) in enumerate(zip(house_features, predictions)):
    dot_result = np.dot(house, weights)
    print(f"집{i+1}: {house} · {weights} + {bias} = {dot_result:.1f} + {bias} = {pred:.1f}백만원")
print()

print("=== 3. 신경망: 모든 층이 내적! ===")
print()

# 간단한 신경망 예제
print("🧠 신경망의 기본 구조:")
print("입력 → [가중치와 내적] → 활성화 함수 → 출력")
print()

# 입력 데이터 (배치 크기: 2, 특성: 3)
X = np.array([
    [1, 2, 3],    # 샘플 1
    [4, 5, 6]     # 샘플 2
])

# 첫 번째 층 가중치 (3개 입력 → 4개 뉴런)
W1 = np.random.randn(3, 4) * 0.5
b1 = np.random.randn(4) * 0.1

# 두 번째 층 가중치 (4개 입력 → 2개 뉴런) 
W2 = np.random.randn(4, 2) * 0.5
b2 = np.random.randn(2) * 0.1

print(f"입력 X shape: {X.shape}")
print(f"첫 번째 층 가중치 W1 shape: {W1.shape}")
print(f"두 번째 층 가중치 W2 shape: {W2.shape}")
print()

# 순전파 (내적의 연속!)
def relu(x):
    return np.maximum(0, x)

# 첫 번째 층
Z1 = np.dot(X, W1) + b1  # 내적 + 편향
A1 = relu(Z1)            # 활성화 함수

# 두 번째 층  
Z2 = np.dot(A1, W2) + b2 # 내적 + 편향
A2 = Z2                  # 출력층 (선형)

print("순전파 과정:")
print(f"첫 번째 층: X @ W1 + b1 = {X.shape} @ {W1.shape} → {Z1.shape}")
print(f"활성화 후: {A1.shape}")
print(f"두 번째 층: A1 @ W2 + b2 = {A1.shape} @ {W2.shape} → {Z2.shape}")
print(f"최종 출력: {A2.shape}")
print()

print("=== 4. 코사인 유사도: 추천시스템의 핵심 ===")
print()

# 사용자-아이템 평점 데이터
users = np.array([
    [5, 3, 0, 1, 4],  # 사용자1의 영화 평점
    [4, 0, 0, 1, 3],  # 사용자2  
    [1, 1, 0, 5, 2],  # 사용자3
    [1, 0, 0, 4, 1],  # 사용자4
])

movies = ["액션", "로맨스", "공포", "SF", "코미디"]

print("🎬 영화 추천 시스템:")
print("사용자별 영화 평점 (5점 만점):")
for i, user_ratings in enumerate(users):
    print(f"사용자{i+1}: {user_ratings}")
print(f"영화 장르: {movies}")
print()

# 코사인 유사도 계산
def cosine_similarity(a, b):
    """코사인 유사도 = (a·b) / (|a|×|b|)"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 사용자1과 다른 사용자들의 유사도
user1 = users[0]
print("사용자1과 다른 사용자들의 유사도:")
for i in range(1, len(users)):
    similarity = cosine_similarity(user1, users[i])
    print(f"사용자1 vs 사용자{i+1}: {similarity:.4f}")

# 가장 유사한 사용자 찾기
similarities = [cosine_similarity(user1, users[i]) for i in range(1, len(users))]
most_similar_idx = np.argmax(similarities) + 1
print(f"\n👥 가장 유사한 사용자: 사용자{most_similar_idx + 1}")
print()

print("=== 5. 주성분 분석(PCA): 차원축소 ===")
print()

# 2차원 데이터 생성
np.random.seed(42)
# 상관관계가 있는 데이터 생성
n_samples = 100
X_original = np.random.randn(n_samples, 2)
# 회전 변환으로 상관관계 만들기
rotation_matrix = np.array([[1, 0.8], [0.8, 1]])
X_corr = X_original @ rotation_matrix

print("📊 PCA로 차원 축소:")
print(f"원본 데이터 shape: {X_corr.shape}")

# 데이터 중심화
X_centered = X_corr - np.mean(X_corr, axis=0)

# 공분산 행렬 (내적 사용!)
cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
print(f"공분산 행렬 계산: X.T @ X / (n-1)")

# 고유값, 고유벡터
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 가장 큰 고유값에 해당하는 주성분
pc1 = eigenvectors[:, np.argmax(eigenvalues)]
print(f"첫 번째 주성분: {pc1}")

# 데이터를 주성분에 투영 (내적!)
projected_data = np.dot(X_centered, pc1)
print(f"투영된 데이터 shape: {projected_data.shape} (2D → 1D)")
print()

print("=== 6. 어텐션 메커니즘: 현대 AI의 핵심 ===")
print()

# 간단한 어텐션 예제
print("🤖 트랜스포머의 어텐션 메커니즘:")
print("Query, Key, Value 모두 내적 연산!")
print()

# 문장: "I love machine learning"
seq_len, d_model = 4, 8  # 시퀀스 길이, 모델 차원

# Query, Key, Value 행렬 (임의 생성)
Q = np.random.randn(seq_len, d_model) * 0.1  # Query
K = np.random.randn(seq_len, d_model) * 0.1  # Key  
V = np.random.randn(seq_len, d_model) * 0.1  # Value

print(f"Query shape: {Q.shape}")
print(f"Key shape: {K.shape}")
print(f"Value shape: {V.shape}")

# 어텐션 점수 계산 (Q와 K의 내적!)
attention_scores = np.dot(Q, K.T) / np.sqrt(d_model)
print(f"\n어텐션 점수 = Q @ K.T / √d_model")
print(f"어텐션 점수 shape: {attention_scores.shape}")

# 소프트맥스로 정규화
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(attention_scores)
print(f"어텐션 가중치 shape: {attention_weights.shape}")

# Value와 가중합 (또 내적!)
output = np.dot(attention_weights, V)
print(f"최종 출력 = attention_weights @ V")
print(f"최종 출력 shape: {output.shape}")
print()

print("=== 7. 내적의 기하학적 의미 ===")
print()

# 벡터 유사도 예제
vec1 = np.array([1, 0])    # 수평 벡터
vec2 = np.array([1, 1])    # 45도 벡터
vec3 = np.array([0, 1])    # 수직 벡터
vec4 = np.array([-1, 0])   # 반대 벡터

vectors = [vec1, vec2, vec3, vec4]
labels = ["수평(0°)", "대각선(45°)", "수직(90°)", "반대(180°)"]

print("📐 벡터 간 각도와 내적의 관계:")
print("기준 벡터:", vec1)
print()

for vec, label in zip(vectors, labels):
    dot_prod = np.dot(vec1, vec)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec)
    cos_sim = dot_prod / (norm1 * norm2)
    angle = np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi
    
    print(f"{label:12}: 내적={dot_prod:4.1f}, 코사인유사도={cos_sim:5.2f}, 각도={angle:5.1f}°")

print()
print("💡 해석:")
print("- 내적 > 0: 같은 방향 (각도 < 90°)")
print("- 내적 = 0: 수직 (각도 = 90°)")  
print("- 내적 < 0: 반대 방향 (각도 > 90°)")
print()

print("=== 8. 실전 머신러닝 파이프라인에서 내적 ===")
print()

# 완전한 예제: 간단한 분류기
print("🔍 실전 예제: 꽃 분류기")
print()

# 가상의 꽃 데이터 (꽃잎 길이, 꽃잎 너비, 꽃받침 길이, 꽃받침 너비)
flower_data = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 꽃 1
    [4.9, 3.0, 1.4, 0.2],  # 꽃 2  
    [6.2, 3.4, 5.4, 2.3],  # 꽃 3
    [5.9, 3.0, 5.1, 1.8],  # 꽃 4
])

labels = np.array([0, 0, 1, 1])  # 0: setosa, 1: virginica

print("꽃 데이터 (꽃잎길이, 꽃잎너비, 꽃받침길이, 꽃받침너비):")
flower_names = ["setosa", "setosa", "virginica", "virginica"]
for i, (data, name) in enumerate(zip(flower_data, flower_names)):
    print(f"꽃{i+1} ({name:9}): {data}")
print()

# 간단한 선형 분류기 학습 (내적 기반!)
def train_linear_classifier(X, y, learning_rate=0.01, epochs=100):
    """경사하강법으로 선형 분류기 학습"""
    n_features = X.shape[1]
    weights = np.random.randn(n_features) * 0.01
    bias = 0
    
    for epoch in range(epochs):
        # 순전파 (내적!)
        z = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-z))  # 시그모이드
        
        # 손실 계산
        loss = -np.mean(y * np.log(predictions + 1e-7) + (1-y) * np.log(1-predictions + 1e-7))
        
        # 역전파 (그래디언트 계산에도 내적!)
        dz = predictions - y
        dw = np.dot(X.T, dz) / len(X)  # 내적으로 그래디언트 계산
        db = np.mean(dz)
        
        # 가중치 업데이트
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    return weights, bias

print("분류기 학습 과정:")
weights, bias = train_linear_classifier(flower_data, labels)
print(f"\n학습된 가중치: {weights}")
print(f"학습된 편향: {bias:.4f}")
print()

# 예측 (내적 사용!)
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias  # 내적으로 예측!
    return 1 / (1 + np.exp(-z))

predictions = predict(flower_data, weights, bias)
print("예측 결과:")
for i, (pred, actual, name) in enumerate(zip(predictions, labels, flower_names)):
    pred_class = "virginica" if pred > 0.5 else "setosa"
    print(f"꽃{i+1}: 예측확률={pred:.3f} → 예측클래스={pred_class:9} (실제: {name})")
print()

print("=== 9. 내적이 중요한 이유 정리 ===")
print()

print("🎯 머신러닝에서 내적의 역할:")
print()
print("1️⃣ 선형 변환의 기초")
print("   - 모든 선형 모델의 핵심 연산")
print("   - 특성과 가중치의 결합")
print()
print("2️⃣ 유사도 측정")
print("   - 코사인 유사도로 데이터 간 관계 파악")
print("   - 추천 시스템, 검색 엔진의 기초")
print()
print("3️⃣ 차원 축소")
print("   - PCA에서 주성분 투영")
print("   - 고차원 → 저차원 변환")
print()
print("4️⃣ 신경망의 모든 층")
print("   - 입력과 가중치의 선형 결합")
print("   - 딥러닝의 기본 빌딩 블록")
print()
print("5️⃣ 어텐션 메커니즘")
print("   - Query, Key 간의 관련성 계산")
print("   - 트랜스포머, BERT, GPT의 핵심")
print()

print("⚡ 성능상 장점:")
print("- 벡터화 연산으로 초고속 처리")
print("- GPU/TPU 가속 최적화")
print("- 메모리 효율적 계산")
print("- 수치적 안정성")
print()

print("🚀 결론:")
print("내적은 머신러닝의 '심장'입니다!")
print("- 모든 예측이 내적으로 시작")
print("- 모든 학습이 내적으로 진행")  
print("- 모든 유사도가 내적으로 계산")
print("- 모든 변환이 내적으로 수행")
print()
print("💡 내적을 이해하면 머신러닝이 보입니다!")