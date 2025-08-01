import numpy as np
import matplotlib.pyplot as plt

print("=== 벡터란 무엇인가? ===")
print()

print("🤔 '움직이고 있는 데이터가 벡터야?'라는 질문에 대한 답:")
print()

print("벡터는 '움직임' 그 자체가 아니라,")
print("'방향과 크기를 가진 데이터 묶음'입니다!")
print()

print("=== 1. 벡터의 정확한 정의 ===")
print()

print("📊 벡터 = 순서가 있는 숫자들의 리스트")
print("예시: [3, 4] = 3과 4라는 두 개의 값")
print()

# 벡터 예제
vector_2d = np.array([3, 4])
print(f"2차원 벡터: {vector_2d}")
print(f"- 첫 번째 성분: {vector_2d[0]} (x 방향)")
print(f"- 두 번째 성분: {vector_2d[1]} (y 방향)")
print(f"- 벡터의 크기: {np.linalg.norm(vector_2d):.2f}")
print()

print("=== 2. 벡터를 바라보는 다양한 관점 ===")
print()

print("🎯 관점 1: 수학적 관점")
print("벡터 = 원점에서 특정 점까지의 '방향'과 '거리'")
print("       ↗ (3,4)")
print("   원점(0,0)")
print()

print("🎯 관점 2: 데이터 관점") 
print("벡터 = 여러 개의 특성(feature)을 묶은 데이터")
student_vector = np.array([85, 90, 78])
print(f"학생 성적 벡터: {student_vector}")
print("→ [수학점수, 영어점수, 과학점수]")
print()

print("🎯 관점 3: 머신러닝 관점")
print("벡터 = 하나의 샘플(sample)을 표현하는 숫자들")
house_vector = np.array([100, 3, 2, 15])
print(f"주택 정보 벡터: {house_vector}")
print("→ [면적, 방개수, 층수, 연식]")
print()

print("=== 3. '움직임'과 벡터의 관계 ===")
print()

print("🏃 움직임을 벡터로 표현하기:")
print()

# 위치와 속도 벡터
position_start = np.array([0, 0])
position_end = np.array([3, 4])
velocity = position_end - position_start

print(f"시작 위치: {position_start}")
print(f"끝 위치: {position_end}")
print(f"이동 벡터: {velocity}")
print("→ '3만큼 오른쪽, 4만큼 위쪽으로 이동'")
print()

print("⚽ 실제 움직임 예제:")
# 축구공의 움직임
ball_positions = np.array([
    [0, 0],    # 시작
    [2, 3],    # 1초 후
    [5, 4],    # 2초 후
    [8, 2],    # 3초 후
])

print("축구공의 위치 변화:")
for i, pos in enumerate(ball_positions):
    print(f"{i}초: 위치 {pos}")

# 속도 벡터 계산
velocities = []
for i in range(len(ball_positions)-1):
    velocity = ball_positions[i+1] - ball_positions[i]
    velocities.append(velocity)
    print(f"{i}→{i+1}초 속도 벡터: {velocity}")

print()

print("=== 4. 데이터에서 벡터의 의미 ===")
print()

print("📈 정적인 데이터도 벡터입니다:")
print()

# 여러 종류의 벡터 데이터
examples = {
    "학생 정보": [20, 175, 70, 85],  # 나이, 키, 몸무게, 성적
    "영화 정보": [120, 8.5, 2019, 1500],  # 러닝타임, 평점, 개봉년도, 관객수(만명)
    "주식 정보": [50000, 52000, 48000, 51000, 1000000],  # 시가, 고가, 저가, 종가, 거래량
    "이미지 픽셀": [255, 128, 64, 32, 0],  # RGB 값들
}

for name, data in examples.items():
    vector = np.array(data)
    print(f"{name}: {vector}")
    print(f"  → 차원: {len(vector)}차원")
    print(f"  → 크기: {np.linalg.norm(vector):.2f}")
    print()

print("💡 핵심 깨달음:")
print("벡터는 '움직이는 것'이 아니라 '여러 정보를 담는 상자'입니다!")
print()

print("=== 5. 시계열 데이터 vs 벡터 ===")
print()

print("🕐 시계열 데이터 (시간에 따라 변하는 데이터):")
time_series = [100, 105, 103, 108, 112, 109, 115]
print(f"주식 가격: {time_series}")
print("→ 각 시점의 가격을 나열한 것")
print()

print("📊 이를 벡터로 표현:")
price_vector = np.array(time_series)
print(f"가격 벡터: {price_vector}")
print("→ 7차원 벡터 (7개의 시점 정보)")
print()

print("🔄 움직임의 패턴을 벡터로:")
# 주가 변화량 벡터
price_changes = np.diff(time_series)
print(f"가격 변화 벡터: {price_changes}")
print("→ 각 시점에서의 변화량")
print()

print("=== 6. 벡터의 다양한 해석 ===")
print()

# 같은 벡터, 다른 의미
mystery_vector = np.array([3, 4])

interpretations = [
    "수학: 원점에서 (3,4) 지점으로의 방향과 거리",
    "물리: 3m 오른쪽, 4m 위쪽으로의 이동",
    "게임: 캐릭터의 x=3, y=4 위치",
    "ML: 어떤 물체의 두 가지 특성 값",
    "이미지: 2개 픽셀의 밝기 값",
    "경제: 두 상품의 가격 정보"
]

print(f"벡터 {mystery_vector}의 다양한 해석:")
for i, interpretation in enumerate(interpretations, 1):
    print(f"{i}. {interpretation}")
print()

print("=== 7. 실제 머신러닝에서 벡터 ===")
print()

print("🤖 머신러닝에서 벡터는 '정적인 정보 묶음':")
print()

# 이미지 벡터
print("1️⃣ 이미지 데이터:")
image_vector = np.random.randint(0, 256, 784)  # 28x28 이미지를 펼친 것
print(f"이미지 벡터 크기: {image_vector.shape}")
print("→ 784개 픽셀 값을 일렬로 나열")
print("→ '움직이지' 않는 정적인 정보")
print()

print("2️⃣ 텍스트 데이터:")
# 단어 빈도 벡터
text_vector = np.array([5, 3, 0, 2, 1, 0, 4])
vocabulary = ["사과", "바나나", "포도", "딸기", "오렌지", "키위", "수박"]
print("문서의 단어 빈도 벡터:")
for word, count in zip(vocabulary, text_vector):
    if count > 0:
        print(f"  {word}: {count}번")
print("→ 문서의 '정적인' 특성을 숫자로 표현")
print()

print("3️⃣ 센서 데이터:")
sensor_vector = np.array([25.3, 60.2, 1013.25, 45.1])
sensor_names = ["온도(°C)", "습도(%)", "기압(hPa)", "풍속(km/h)"]
print("센서 측정값 벡터:")
for name, value in zip(sensor_names, sensor_vector):
    print(f"  {name}: {value}")
print("→ 특정 순간의 '스냅샷'")
print()

print("=== 8. 벡터 연산의 의미 ===")
print()

print("➕ 벡터 덧셈:")
vec1 = np.array([1, 2])
vec2 = np.array([3, 1])
sum_vec = vec1 + vec2
print(f"{vec1} + {vec2} = {sum_vec}")
print("→ 두 이동을 합친 전체 이동")
print("→ 두 특성을 합친 새로운 특성")
print()

print("✖️ 스칼라 곱셈:")
scaled_vec = vec1 * 3
print(f"{vec1} × 3 = {scaled_vec}")
print("→ 같은 방향으로 3배 더 멀리")
print("→ 모든 특성을 3배로 증폭")
print()

print("📐 내적 (Dot Product):")
dot_result = np.dot(vec1, vec2)
print(f"{vec1} · {vec2} = {dot_result}")
print("→ 두 벡터의 '유사도' 측정")
print("→ 두 데이터의 '관련성' 계산")
print()

print("=== 9. 일상 속 벡터 찾기 ===")
print()

everyday_vectors = {
    "쇼핑 목록": [2, 3, 1, 5],  # 사과, 바나나, 우유, 빵 개수
    "성적표": [85, 90, 78, 92],  # 수학, 영어, 과학, 사회 점수
    "건강 데이터": [70, 175, 120, 80],  # 몸무게, 키, 혈압(상), 혈압(하)
    "GPS 위치": [37.5665, 126.9780],  # 위도, 경도
    "RGB 색상": [255, 128, 64],  # 빨강, 초록, 파랑 값
}

print("📱 일상에서 만나는 벡터들:")
for name, data in everyday_vectors.items():
    print(f"{name}: {data}")
print()

print("=== 10. 벡터에 대한 오해와 진실 ===")
print()

print("❌ 흔한 오해들:")
print("1. '벡터는 움직이는 것이다' → 벡터는 정적인 데이터 묶음")
print("2. '벡터는 항상 2D/3D다' → 차원 제한 없음 (1차원~수천차원)")
print("3. '벡터는 화살표다' → 화살표는 벡터의 시각화 방법 중 하나")
print("4. '벡터는 물리학에서만 쓴다' → 모든 데이터 분야에서 활용")
print()

print("✅ 벡터의 진실:")
print("1. 벡터 = 순서가 있는 숫자들의 리스트")
print("2. 벡터 = 다차원 정보를 표현하는 방법")
print("3. 벡터 = 데이터 분석의 기본 단위") 
print("4. 벡터 = 머신러닝의 입력과 출력 형태")
print()

print("=== 11. 실습: 벡터 만들어보기 ===")
print()

print("🏠 당신의 집을 벡터로 표현해봅시다:")
house_features = {
    "면적(㎡)": 85,
    "방 개수": 3,
    "층수": 2,
    "건축년도": 2010,
    "역까지 거리(분)": 5
}

house_vector = np.array(list(house_features.values()))
print("집 정보 벡터:")
for feature, value in house_features.items():
    print(f"  {feature}: {value}")
print(f"벡터 형태: {house_vector}")
print(f"차원: {len(house_vector)}차원")
print()

print("👤 당신을 벡터로 표현해봅시다:")
person_features = {
    "나이": 25,
    "키(cm)": 170,
    "몸무게(kg)": 65,
    "수면시간(시간)": 7,
    "운동횟수(주당)": 3
}

person_vector = np.array(list(person_features.values()))
print("개인 정보 벡터:")
for feature, value in person_features.items():
    print(f"  {feature}: {value}")
print(f"벡터 형태: {person_vector}")
print()

print("🎯 결론:")
print()
print("벡터는 '움직이는 데이터'가 아니라")
print("'여러 정보를 하나로 묶어 표현하는 방법'입니다!")
print()
print("📦 벡터 = 데이터의 '상자'")
print("- 상자 안에는 순서대로 정리된 숫자들")
print("- 각 숫자는 특정한 의미를 가짐")
print("- 상자 전체가 하나의 '객체'나 '샘플'을 나타냄")
print()
print("🚀 이제 벡터를 올바르게 이해했습니다!")
print("데이터 사이언스와 머신러닝에서 벡터는")
print("모든 정보를 수치화하여 컴퓨터가 처리할 수 있게 하는")
print("가장 기본적이고 중요한 형태입니다!")