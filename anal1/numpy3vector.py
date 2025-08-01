import numpy as np
import time

print("=== 벡터화(Vectorization)란 무엇인가? ===")
print()

print("🎯 벡터화의 정의:")
print("반복문을 사용하지 않고 배열 전체에 연산을 한 번에 적용하는 기법")
print("내부적으로는 C 언어로 구현된 최적화된 코드가 실행됨")
print()

print("=== 1. 기본 예제: 벡터화 vs 반복문 ===")

# 데이터 준비
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
numpy_array = np.array(data)

print(f"원본 데이터: {data}")
print()

# 방법 1: 전통적인 반복문 (비벡터화)
print("❌ 비벡터화 방법 (반복문 사용):")
result_loop = []
for i in range(len(data)):
    result_loop.append(data[i] * 2)  # 각 요소에 2를 곱함
print(f"결과: {result_loop}")
print("→ 각 요소를 하나씩 처리 (느림)")
print()

# 방법 2: 벡터화 방법
print("✅ 벡터화 방법:")
result_vectorized = numpy_array * 2  # 모든 요소에 동시에 2를 곱함
print(f"결과: {result_vectorized}")
print("→ 모든 요소를 한 번에 처리 (빠름)")
print()

print("=== 2. 내부 동작 원리 ===")
print()

print("🔍 비벡터화 (반복문)의 내부 동작:")
print("""
for i in range(len(data)):
    temp = data[i]      # 1. 메모리에서 값 읽기
    result = temp * 2   # 2. 연산 수행
    result_list.append(result)  # 3. 결과 저장
    # 위 과정을 n번 반복 (Python 인터프리터 오버헤드)
""")

print("🚀 벡터화의 내부 동작:")
print("""
numpy_array * 2  # 1. C 코드로 구현된 함수 호출
                 # 2. 전체 배열을 한 번에 메모리에서 읽기
                 # 3. SIMD(Single Instruction Multiple Data) 사용
                 # 4. CPU의 여러 코어 활용
                 # 5. 캐시 효율성 극대화
""")

print("=== 3. 성능 비교 실험 ===")

# 큰 배열로 성능 테스트
large_data = list(range(1000000))  # 100만 개 데이터
large_array = np.array(large_data)

print(f"테스트 데이터 크기: {len(large_data):,}개")
print()

# 비벡터화 방법 측정
print("⏱️ 비벡터화 방법 (반복문):")
start_time = time.time()
result_slow = []
for x in large_data:
    result_slow.append(x * 2)
loop_time = time.time() - start_time
print(f"실행 시간: {loop_time:.4f}초")

# 벡터화 방법 측정  
print("\n⚡ 벡터화 방법:")
start_time = time.time()
result_fast = large_array * 2
vector_time = time.time() - start_time
print(f"실행 시간: {vector_time:.6f}초")

# 속도 비교
if vector_time > 0:
    speedup = loop_time / vector_time
    print(f"\n🚀 벡터화가 {speedup:.1f}배 빠름!")
else:
    print(f"\n🚀 벡터화가 측정 불가능할 정도로 빠름!")
print()

print("=== 4. 다양한 벡터화 예제 ===")

# 배열 생성
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

print(f"배열1: {arr1}")
print(f"배열2: {arr2}")
print()

print("1️⃣ 기본 산술 연산:")
print(f"덧셈: {arr1 + arr2}")        # 벡터화된 덧셈
print(f"뺄셈: {arr1 - arr2}")        # 벡터화된 뺄셈
print(f"곱셈: {arr1 * arr2}")        # 벡터화된 곱셈
print(f"나눗셈: {arr2 / arr1}")      # 벡터화된 나눗셈
print()

print("2️⃣ 수학 함수:")
print(f"제곱근: {np.sqrt(arr1)}")    # 벡터화된 제곱근
print(f"지수함수: {np.exp(arr1)}")   # 벡터화된 지수함수
print(f"사인: {np.sin(arr1)}")       # 벡터화된 사인함수
print()

print("3️⃣ 조건부 연산:")
print(f"5보다 큰 값: {arr2 > 5}")        # 벡터화된 비교
print(f"조건부 선택: {np.where(arr1 > 3, arr1, 0)}")  # 벡터화된 조건 선택
print()

print("=== 5. 브로드캐스팅과 벡터화 ===")

# 2차원 배열
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6]])
scalar = 10
vector = np.array([1, 2, 3])

print("2D 배열:")
print(matrix)
print(f"스칼라: {scalar}")
print(f"1D 배열: {vector}")
print()

print("벡터화된 브로드캐스팅:")
print("스칼라 덧셈:")
print(matrix + scalar)  # 모든 요소에 10을 더함
print()

print("1D 배열 덧셈:")
print(matrix + vector)  # 각 행에 [1, 2, 3]을 더함
print()

print("=== 6. 실무 예제: 이미지 처리 ===")

# 가상의 이미지 데이터 (100x100 픽셀)
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

print(f"이미지 크기: {image.shape}")
print(f"원본 픽셀 값 범위: {image.min()} ~ {image.max()}")

# 벡터화된 이미지 처리
print("\n벡터화된 이미지 처리:")

# 1. 밝기 조절 (모든 픽셀에 동시 적용)
bright_image = np.clip(image + 50, 0, 255)
print(f"밝기 조절 후: {bright_image.min()} ~ {bright_image.max()}")

# 2. 대비 조절 (모든 픽셀에 동시 적용)
contrast_image = np.clip(image * 1.5, 0, 255)
print(f"대비 조절 후: {contrast_image.min()} ~ {contrast_image.max()}")

# 3. 임계값 처리 (모든 픽셀에 동시 적용)
threshold_image = np.where(image > 128, 255, 0)
print(f"임계값 처리 후: {np.unique(threshold_image)}")
print()

print("=== 7. 실무 예제: 데이터 전처리 ===")

# 학생 성적 데이터
scores = np.array([
    [85, 90, 78],  # 학생1: 수학, 영어, 과학
    [92, 88, 95],  # 학생2
    [76, 82, 89],  # 학생3
    [88, 94, 91],  # 학생4
    [79, 85, 87]   # 학생5
])

print("원본 성적 데이터:")
print(scores)
print()

# 벡터화된 데이터 전처리
print("벡터화된 데이터 전처리:")

# 1. 정규화 (0-1 범위)
min_scores = scores.min(axis=0)
max_scores = scores.max(axis=0)
normalized = (scores - min_scores) / (max_scores - min_scores)
print("정규화 (0-1 범위):")
print(normalized)
print()

# 2. 표준화 (평균=0, 표준편차=1)
mean_scores = scores.mean(axis=0)
std_scores = scores.std(axis=0)
standardized = (scores - mean_scores) / std_scores
print("표준화 (Z-score):")
print(standardized)
print()

# 3. 보너스 점수 추가
bonus = np.array([5, 3, 7])  # 각 과목별 보너스
scores_with_bonus = scores + bonus
print(f"보너스 점수 {bonus} 추가:")
print(scores_with_bonus)
print()

print("=== 8. 벡터화의 핵심 원리 ===")
print()

print("🔧 SIMD (Single Instruction Multiple Data):")
print("- CPU가 하나의 명령으로 여러 데이터를 동시 처리")
print("- 예: 4개 숫자를 동시에 더하기")
print()

print("💾 메모리 접근 최적화:")
print("- 연속된 메모리 블록을 한 번에 읽기")
print("- CPU 캐시 효율성 극대화")
print("- 메모리 대역폭 최대 활용")
print()

print("⚡ 컴파일된 C 코드:")
print("- Python 인터프리터 오버헤드 제거")
print("- 최적화된 어셈블리 코드 실행")
print("- 컴파일러 최적화 혜택")
print()

print("=== 9. 언제 벡터화를 사용할까? ===")
print()

print("✅ 벡터화 사용 권장:")
print("1. 배열의 모든 요소에 같은 연산")
print("2. 수학적 계산 (통계, 선형대수)")
print("3. 이미지/신호 처리")
print("4. 머신러닝 연산")
print("5. 대용량 데이터 처리")
print()

print("❌ 벡터화 사용 어려운 경우:")
print("1. 복잡한 조건문이 많은 경우")
print("2. 순서에 의존적인 연산")
print("3. 각 요소마다 다른 처리가 필요한 경우")
print("4. 외부 API 호출이 필요한 경우")
print()

print("=== 10. 실전 팁 ===")
print()

print("💡 성능 최적화 팁:")
print("1. 가능한 모든 연산을 벡터화")
print("2. 적절한 데이터 타입 선택 (float32 vs float64)")
print("3. 메모리 레이아웃 고려 (연속된 배열 사용)")
print("4. 브로드캐스팅 활용")
print()

print("🚨 주의사항:")
print("1. 메모리 사용량 증가 가능")
print("2. 작은 배열에서는 오버헤드가 클 수 있음")
print("3. 디버깅이 어려울 수 있음")
print()

# 벡터화 vs 비벡터화 코드 비교
print("=== 코드 비교 예제 ===")
print()

print("❌ 비벡터화 코드:")
print("""
result = []
for i in range(len(data)):
    if data[i] > threshold:
        result.append(data[i] ** 2)
    else:
        result.append(0)
""")

print("✅ 벡터화 코드:")
print("""
result = np.where(data > threshold, data ** 2, 0)
""")

print()
print("🎯 결론:")
print("벡터화는 NumPy의 핵심 기능으로,")
print("반복문 없이 배열 전체에 연산을 적용하여")
print("- 10-100배 빠른 속도")
print("- 간결한 코드")  
print("- 메모리 효율성")
print("을 제공합니다!")
print()
print("🚀 데이터 사이언스와 머신러닝에서 필수적인 개념입니다!")