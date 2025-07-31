import numpy as np
import matplotlib.pyplot as plt

print("=== NumPy 기본 배열 생성 ===")
# 단위행렬 (Identity Matrix)
f = np.eye(2)  # 2x2 단위 행렬
print("2x2 단위행렬:")
print(f)
print()

print("=== 다양한 난수 생성 방법 ===")

# 1. 균등분포 난수 (0~1 사이)
print("1. np.random.rand() - 균등분포 [0, 1)")
uniform_random = np.random.rand(5)
print(f"균등분포 난수 5개: {uniform_random}")
print(f"최솟값: {np.min(uniform_random):.4f}, 최댓값: {np.max(uniform_random):.4f}")
print()

# 2. 정규분포 난수 (평균=0, 표준편차=1)
print("2. np.random.randn() - 표준정규분포 N(0,1)")
normal_random = np.random.randn(5)
print(f"정규분포 난수 5개: {normal_random}")
print(f"최솟값: {np.min(normal_random):.4f}, 최댓값: {np.max(normal_random):.4f}")
print("※ 음수도 나올 수 있음 (평균=0이므로)")
print()

# 3. 정수 난수
print("3. np.random.randint() - 정수 난수")
int_random = np.random.randint(1, 10, size=(2, 3))
print(f"1~9 사이 정수로 2x3 배열:")
print(int_random)
print()

# 4. 다차원 정규분포
print("4. 다차원 정규분포")
normal_2d = np.random.randn(2, 3)
print(f"표준정규분포 2x3 배열:")
print(normal_2d)
print()

print("=== 난수 시드(Seed)의 중요성 ===")

# 시드 없이 난수 생성 (매번 다름)
print("시드 설정 전 - 매번 다른 결과:")
for i in range(3):
    print(f"실행 {i+1}: {np.random.rand(3)}")

print()

# 시드 설정으로 재현 가능한 난수
print("시드 설정 후 - 항상 같은 결과:")
for i in range(3):
    np.random.seed(42)  # 같은 시드 설정
    print(f"실행 {i+1}: {np.random.rand(3)}")

print()

print("=== 시드가 필요한 이유 ===")
print("1. 실험 재현성: 연구 결과를 다른 사람이 재현할 수 있음")
print("2. 디버깅: 같은 조건에서 테스트 가능")
print("3. 모델 비교: 동일한 데이터로 알고리즘 성능 비교")
print("4. 교육: 강의나 튜토리얼에서 같은 결과 보장")
print()

# 실제 활용 예제
print("=== 실제 활용 예제 ===")

# 1. 머신러닝 데이터셋 분할
np.random.seed(123)
print("1. 데이터셋 분할 (train/test)")
data_size = 1000
indices = np.random.permutation(data_size)  # 인덱스 셔플
train_size = int(0.8 * data_size)
train_indices = indices[:train_size]
test_indices = indices[train_size:]
print(f"훈련 데이터: {len(train_indices)}개, 테스트 데이터: {len(test_indices)}개")
print()

# 2. 가중치 초기화 (딥러닝)
print("2. 신경망 가중치 초기화")
np.random.seed(456)
# Xavier 초기화
input_size, output_size = 784, 128
weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
print(f"가중치 행렬 크기: {weights.shape}")
print(f"가중치 평균: {np.mean(weights):.6f}, 표준편차: {np.std(weights):.6f}")
print()

# 3. 몬테카를로 시뮬레이션
print("3. 몬테카를로 π 추정")
np.random.seed(789)
n_samples = 100000
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)
inside_circle = (x**2 + y**2) <= 1
pi_estimate = 4 * np.mean(inside_circle)
print(f"π 추정값: {pi_estimate:.4f} (실제: {np.pi:.4f})")
print()

print("=== 다양한 분포에서 난수 생성 ===")

# 다양한 분포 함수들
np.random.seed(100)

print("1. 균등분포 - 사용자 정의 범위")
custom_uniform = np.random.uniform(5, 15, 5)
print(f"5~15 사이 균등분포: {custom_uniform}")

print("\n2. 정규분포 - 사용자 정의 평균, 표준편차")
custom_normal = np.random.normal(100, 15, 5)  # 평균=100, 표준편차=15
print(f"평균=100, 표준편차=15 정규분포: {custom_normal}")

print("\n3. 포아송 분포")
poisson_data = np.random.poisson(3, 5)  # 평균=3
print(f"포아송 분포 (λ=3): {poisson_data}")

print("\n4. 지수분포")
exponential_data = np.random.exponential(2, 5)  # 평균=2
print(f"지수분포 (λ=2): {exponential_data}")

print("\n5. 베타분포")
beta_data = np.random.beta(2, 5, 5)  # α=2, β=5
print(f"베타분포 (α=2, β=5): {beta_data}")
print()

print("=== 실무에서 자주 사용하는 패턴 ===")

# 1. 노이즈 추가
print("1. 데이터에 노이즈 추가")
np.random.seed(200)
clean_signal = np.sin(np.linspace(0, 2*np.pi, 50))
noise = np.random.normal(0, 0.1, 50)  # 평균=0, 표준편차=0.1 노이즈
noisy_signal = clean_signal + noise
print(f"원본 신호 범위: [{np.min(clean_signal):.2f}, {np.max(clean_signal):.2f}]")
print(f"노이즈 추가 후: [{np.min(noisy_signal):.2f}, {np.max(noisy_signal):.2f}]")

# 2. 부트스트랩 샘플링
print("\n2. 부트스트랩 샘플링")
np.random.seed(300)
original_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bootstrap_sample = np.random.choice(original_data, size=10, replace=True)
print(f"원본 데이터: {original_data}")
print(f"부트스트랩 샘플: {bootstrap_sample}")

# 3. 데이터 증강 (이미지 처리에서 자주 사용)
print("\n3. 데이터 증강 시뮬레이션")
np.random.seed(400)
# 회전각도 랜덤 생성
rotation_angles = np.random.uniform(-15, 15, 5)  # -15도 ~ +15도
# 밝기 조절 팩터
brightness_factors = np.random.uniform(0.8, 1.2, 5)  # 80% ~ 120%
print(f"랜덤 회전각: {rotation_angles}")
print(f"밝기 조절: {brightness_factors}")
print()

print("=== 시드 관리 팁 ===")
print("1. 전역 시드: np.random.seed() - 모든 난수 함수에 영향")
print("2. 지역 시드: RandomState 객체 사용 - 독립적인 난수 생성기")
print("3. 최신 방법: Generator 객체 사용 (NumPy 1.17+)")
print()

# RandomState 사용 예제
print("RandomState 사용 예제:")
rng1 = np.random.RandomState(42)
rng2 = np.random.RandomState(42)

print(f"rng1: {rng1.rand(3)}")
print(f"rng2: {rng2.rand(3)}")  # 같은 결과
print()

# 최신 Generator 사용 예제
print("Generator 사용 예제 (권장):")
rng = np.random.default_rng(42)  # 새로운 방식
random_data = rng.random(5)
normal_data = rng.normal(0, 1, 5)
print(f"Generator 균등분포: {random_data}")
print(f"Generator 정규분포: {normal_data}")

print("\n=== 딥러닝에서 난수가 중요한 이유 ===")
print("1. 가중치 초기화: 신경망 학습의 시작점")
print("2. 드롭아웃: 과적합 방지를 위한 랜덤 뉴런 비활성화")
print("3. 데이터 셔플링: 배치 단위 학습 시 데이터 순서 섞기")
print("4. 데이터 증강: 훈련 데이터 다양성 증가")
print("5. 몬테카를로 방법: 불확실성 추정")

# 간단한 시각화
if True:  # 시각화 코드
    print("\n=== 분포 시각화 ===")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    np.random.seed(42)
    
    # 균등분포
    uniform_data = np.random.rand(1000)
    axes[0,0].hist(uniform_data, bins=30, alpha=0.7, color='blue')
    axes[0,0].set_title('균등분포 [0,1)')
    axes[0,0].set_ylabel('빈도')
    
    # 정규분포
    normal_data = np.random.randn(1000)
    axes[0,1].hist(normal_data, bins=30, alpha=0.7, color='red')
    axes[0,1].set_title('표준정규분포 N(0,1)')
    
    # 정수 난수
    int_data = np.random.randint(1, 7, 1000)  # 주사위
    axes[1,0].hist(int_data, bins=6, alpha=0.7, color='green')
    axes[1,0].set_title('주사위 (정수 난수)')
    axes[1,0].set_ylabel('빈도')
    axes[1,0].set_xlabel('값')
    
    # 포아송 분포
    poisson_data = np.random.poisson(3, 1000)
    axes[1,1].hist(poisson_data, bins=15, alpha=0.7, color='orange')
    axes[1,1].set_title('포아송분포 (λ=3)')
    axes[1,1].set_xlabel('값')
    
    plt.tight_layout()
    plt.savefig('random_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("분포 그래프가 생성되었습니다!")