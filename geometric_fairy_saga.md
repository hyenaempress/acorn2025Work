

## 🌟 서막: 지오메트리아 요정 왕국의 전설

*오래전, 2차원과 3차원이 교차하는 신비한 땅 '지오메트리아'가 있었다...*

```
🌸 고대 요정 서적에 기록된 예언:

"점들이 춤추고 선분들이 노래하는 세상에서,
어떤 이는 가장 가까운 친구를 찾고,
어떤 이는 모든 점을 품는 껍질을 만들며,
어떤 이는 선분들의 만남을 예언할 것이다.

하지만 진정한 기하 요정은
공간의 차원을 넘나들며 모든 형태의 아름다움을 
이해하는 자이리라."
```

---

## 🏛️ 전체 요정 캐스트 소개

| 요정 | ⏰ 시간복잡도 | 💫 마법 속성 | 소속 | 역할 | 특기 |
|--------|-------------|-------------|------|------|------|
| 🔍 **포인티** | **O(n²)** | **거리 마법** | **기본 요정단** | **탐지사** | **가장 가까운 점 찾기** |
| 🌊 **컨벡시** | **O(n log n)** | **껍질 마법** | **기본 요정단** | **수호자** | **볼록 껍질 생성** |
| ⚡ **인터섹** | **O(n log n)** | **교차 마법** | **기하 마법사 길드** | **예언자** | **선분 교차점 탐지** |
| 📐 **트라이앵** | **O(n²)** | **삼각 마법** | **기하 마법사 길드** | **건축가** | **들로네 삼각분할** |
| 🗺️ **보로노이** | **O(n log n)** | **영역 마법** | **영토 관리소** | **영주** | **보로노이 다이어그램** |
| 🎯 **폴리고니** | **O(n)** | **내부 마법** | **영토 관리소** | **판별사** | **점-다각형 내부 판정** |
| 🔄 **로테이터** | **O(1)** | **회전 마법** | **변환 마법 학원** | **춤꾼** | **회전 변환** |
| 📏 **스케일러** | **O(1)** | **크기 마법** | **변환 마법 학원** | **조각가** | **확대/축소** |
| 🌀 **스위퍼** | **O(n log n)** | **회전선 마법** | **고급 기하 협회** | **청소부** | **회전 스위핑** |
| 🔬 **KD트리나** | **O(log n)** | **차원 분할 마법** | **고급 기하 협회** | **도서관장** | **k차원 공간 검색** |

---

## 🌟 제1막: 기본 요정단의 모험

### 🔍 포인티 (Closest Pair) - "가장 가까운 친구를 찾는 요정"

#### 캐릭터 설정 🧚‍♀️
- **클래스**: 탐지사 (Detective)
- **정체**: 수많은 점들 중에서 가장 가까운 두 점을 찾는 따뜻한 마음의 요정
- **별명**: "우정의 탐지사"
- **성격**: 따뜻하고 세심함, 모든 점들 사이의 관계를 관찰
- **특기**: 거리 계산과 최적화로 가장 가까운 쌍 발견
- **철학**: "가장 가까운 거리에 가장 깊은 우정이 있다"
- **말버릇**: "가까운 친구를 찾아보자!"
- **무기**: 거리 측정 마법 지팡이 (유클리드 거리 계산)

#### 가장 가까운 점 쌍 찾기 모험 💫
```python
import math
import matplotlib.pyplot as plt
import random

class 포인티의_우정탐지:
    def __init__(self):
        print("🔍 포인티: '안녕! 가장 가까운 친구들을 찾아주는 포인티야!'")
    
    def 거리계산(self, 점1, 점2):
        """두 점 사이의 유클리드 거리 계산"""
        x1, y1 = 점1
        x2, y2 = 점2
        거리 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return 거리
    
    def 무차별_최근접점_찾기(self, 점들):
        """무차별 대입법으로 가장 가까운 점 쌍 찾기"""
        print(f"🔍 포인티: '총 {len(점들)}개의 점에서 가장 가까운 친구들을 찾아볼게!'")
        
        if len(점들) < 2:
            print("😅 '점이 2개 미만이면 친구를 찾을 수 없어!'")
            return None, float('inf')
        
        최소거리 = float('inf')
        최근접쌍 = None
        비교횟수 = 0
        
        print(f"✨ '모든 점 쌍의 거리를 하나씩 확인해보자!'")
        
        for i in range(len(점들)):
            for j in range(i + 1, len(점들)):
                점1 = 점들[i]
                점2 = 점들[j]
                거리 = self.거리계산(점1, 점2)
                비교횟수 += 1
                
                print(f"   📏 점{i+1}{점1} ↔ 점{j+1}{점2}: 거리 = {거리:.2f}")
                
                if 거리 < 최소거리:
                    최소거리 = 거리
                    최근접쌍 = (점1, 점2)
                    print(f"   🌟 '새로운 최근접 쌍 발견! 거리: {거리:.2f}'")
        
        print(f"\n🎉 포인티: '찾았어! 가장 가까운 친구들!'")
        print(f"   👫 최근접 쌍: {최근접쌍[0]} ↔ {최근접쌍[1]}")
        print(f"   📏 최소 거리: {최소거리:.2f}")
        print(f"   🔢 총 비교 횟수: {비교횟수}")
        
        return 최근접쌍, 최소거리
    
    def 분할정복_최근접점_찾기(self, 점들):
        """분할정복으로 효율적인 최근접점 찾기"""
        print(f"⚡ 포인티: '분할정복 마법으로 더 빠르게 찾아볼게!'")
        
        def 분할정복_도우미(점들_정렬):
            n = len(점들_정렬)
            
            # 기저 사례: 점이 3개 이하면 무차별 대입
            if n <= 3:
                return self.무차별_작은그룹(점들_정렬)
            
            # 중간점으로 분할
            중간 = n // 2
            중간점 = 점들_정렬[중간]
            
            print(f"   ✂️ '{len(점들_정렬)}개 점을 절반으로 나누자!'")
            print(f"      왼쪽: {점들_정렬[:중간]}")
            print(f"      오른쪽: {점들_정렬[중간:]}")
            
            # 재귀적으로 좌우 부분 해결
            왼쪽_쌍, 왼쪽_거리 = 분할정복_도우미(점들_정렬[:중간])
            오른쪽_쌍, 오른쪽_거리 = 분할정복_도우미(점들_정렬[중간:])
            
            # 더 작은 거리 선택
            if 왼쪽_거리 <= 오른쪽_거리:
                최소_쌍, 최소_거리 = 왼쪽_쌍, 왼쪽_거리
            else:
                최소_쌍, 최소_거리 = 오른쪽_쌍, 오른쪽_거리
            
            print(f"   🔍 '중앙선 근처에서 더 가까운 쌍이 있는지 확인!'")
            
            # 중앙선 근처의 점들 확인
            중앙선_x = 중간점[0]
            중앙선_후보들 = []
            
            for 점 in 점들_정렬:
                if abs(점[0] - 중앙선_x) < 최소_거리:
                    중앙선_후보들.append(점)
            
            # y좌표 기준으로 정렬
            중앙선_후보들.sort(key=lambda p: p[1])
            
            # 중앙선 근처에서 최근접 쌍 찾기
            for i in range(len(중앙선_후보들)):
                j = i + 1
                while (j < len(중앙선_후보들) and 
                       중앙선_후보들[j][1] - 중앙선_후보들[i][1] < 최소_거리):
                    거리 = self.거리계산(중앙선_후보들[i], 중앙선_후보들[j])
                    if 거리 < 최소_거리:
                        최소_거리 = 거리
                        최소_쌍 = (중앙선_후보들[i], 중앙선_후보들[j])
                        print(f"   🌟 '중앙선에서 더 가까운 쌍 발견! 거리: {거리:.2f}'")
                    j += 1
            
            return 최소_쌍, 최소_거리
        
        # x좌표 기준으로 정렬
        점들_정렬 = sorted(점들, key=lambda p: p[0])
        print(f"   📊 'x좌표 기준으로 정렬: {점들_정렬}'")
        
        결과_쌍, 결과_거리 = 분할정복_도우미(점들_정렬)
        
        print(f"\n🎉 분할정복 결과:")
        print(f"   👫 최근접 쌍: {결과_쌍[0]} ↔ {결과_쌍[1]}")
        print(f"   📏 최소 거리: {결과_거리:.2f}")
        
        return 결과_쌍, 결과_거리
    
    def 무차별_작은그룹(self, 점들):
        """3개 이하 점들에 대한 무차별 대입"""
        최소거리 = float('inf')
        최근접쌍 = None
        
        for i in range(len(점들)):
            for j in range(i + 1, len(점들)):
                거리 = self.거리계산(점들[i], 점들[j])
                if 거리 < 최소거리:
                    최소거리 = 거리
                    최근접쌍 = (점들[i], 점들[j])
        
        return 최근접쌍, 최소거리
    
    def 시각화(self, 점들, 최근접쌍):
        """결과를 시각적으로 표현"""
        print(f"🎨 '결과를 예쁘게 그려볼게!'")
        
        plt.figure(figsize=(10, 8))
        
        # 모든 점 그리기
        x_coords = [p[0] for p in 점들]
        y_coords = [p[1] for p in 점들]
        plt.scatter(x_coords, y_coords, c='lightblue', s=100, alpha=0.7, label='모든 점들')
        
        # 점 번호 표시
        for i, (x, y) in enumerate(점들):
            plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        # 최근접 쌍 강조
        if 최근접쌍:
            점1, 점2 = 최근접쌍
            plt.scatter([점1[0], 점2[0]], [점1[1], 점2[1]], 
                       c='red', s=150, label='최근접 쌍', zorder=5)
            plt.plot([점1[0], 점2[0]], [점1[1], 점2[1]], 
                    'r-', linewidth=3, alpha=0.7, label='최단 거리')
        
        plt.title('🔍 포인티의 가장 가까운 친구 찾기', fontsize=16)
        plt.xlabel('X 좌표')
        plt.ylabel('Y 좌표')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# 실전 예시
print("🔍 포인티의 우정 탐지 모험:")
포인티 = 포인티의_우정탐지()

# 랜덤 점들 생성
random.seed(42)  # 재현 가능한 결과를 위해
점들 = [(random.randint(0, 20), random.randint(0, 20)) for _ in range(8)]

print(f"\n🌟 요정 마을의 점들: {점들}")

print(f"\n" + "="*50)
print("📊 방법 1: 무차별 대입법")
무차별_쌍, 무차별_거리 = 포인티.무차별_최근접점_찾기(점들)

print(f"\n" + "="*50)  
print("📊 방법 2: 분할정복법")
분할정복_쌍, 분할정복_거리 = 포인티.분할정복_최근접점_찾기(점들)

print(f"\n📋 결과 비교:")
print(f"   무차별 대입: {무차별_쌍}, 거리: {무차별_거리:.2f}")
print(f"   분할정복: {분할정복_쌍}, 거리: {분할정복_거리:.2f}")
print(f"   결과 일치: {'✅' if abs(무차별_거리 - 분할정복_거리) < 0.001 else '❌'}")

# 시각화는 matplotlib가 있을 때만
try:
    포인티.시각화(점들, 무차별_쌍)
except:
    print("🎨 '시각화를 위해서는 matplotlib가 필요해!'")
```

**⏰ 시간복잡도**: 
- 무차별 대입: O(n²) - "모든 쌍을 다 확인해야 해!"
- 분할정복: O(n log n) - "똑똑하게 나누어서 정복!"

**💫 마법 속성**: 거리 계산과 최적화를 통한 우정 발견

**🎯 장점**: 확실하고 정확한 결과
**⚠️ 단점**: 무차별 대입은 점이 많아질수록 매우 느려짐

---

### 🌊 컨벡시 (Convex Hull) - "모든 것을 품는 껍질 요정"

#### 캐릭터 설정 🧚‍♂️
- **클래스**: 수호자 (Guardian)
- **정체**: 모든 점들을 포함하는 가장 작은 볼록한 껍질을 만드는 보호 요정
- **별명**: "껍질의 수호자"
- **성격**: 포용력이 크고 보호 본능이 강함, 모든 점을 품고 싶어함
- **특기**: 그라함 스캔과 기프트 래핑으로 볼록 껍질 생성
- **철학**: "모든 점들을 감싸는 가장 작은 울타리가 진정한 보호다"
- **말버릇**: "모두를 안전하게 감싸줄게!"
- **무기**: 볼록 껍질 보호막 (기하학적 경계)

#### 볼록 껍질 생성 모험 🛡️
```python
import math

class 컨벡시의_보호막생성:
    def __init__(self):
        print("🌊 컨벡시: '안녕! 모든 점들을 안전하게 보호하는 컨벡시야!'")
    
    def 외적계산(self, O, A, B):
        """세 점 O, A, B에 대한 외적 계산 (방향 판단)"""
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
    
    def 그라함_스캔(self, 점들):
        """그라함 스캔 알고리즘으로 볼록 껍질 생성"""
        print(f"🌊 컨벡시: '그라함 스캔 마법으로 {len(점들)}개 점의 보호막을 만들어보자!'")
        
        if len(점들) < 3:
            print("😅 '점이 3개 미만이면 볼록 껍질을 만들 수 없어!'")
            return 점들
        
        # 1. 가장 아래쪽 점 찾기 (y가 가장 작고, 같으면 x가 가장 작은 점)
        시작점 = min(점들, key=lambda p: (p[1], p[0]))
        print(f"   🎯 시작점 선택: {시작점} (가장 아래쪽 점)")
        
        # 2. 시작점을 기준으로 각도 순으로 정렬
        def 각도계산(점):
            dx = 점[0] - 시작점[0]
            dy = 점[1] - 시작점[1]
            return math.atan2(dy, dx)
        
        나머지점들 = [p for p in 점들 if p != 시작점]
        정렬된점들 = sorted(나머지점들, key=각도계산)
        
        print(f"   📐 각도 순 정렬: {[시작점] + 정렬된점들}")
        
        # 3. 스택을 사용하여 볼록 껍질 구성
        스택 = [시작점, 정렬된점들[0]]
        print(f"   📚 초기 스택: {스택}")
        
        for i, 현재점 in enumerate(정렬된점들[1:], 2):
            print(f"\n   🔍 단계 {i}: 점 {현재점} 처리")
            print(f"      현재 스택: {스택}")
            
            # 왼쪽으로 회전하지 않는 점들 제거
            while len(스택) > 1:
                외적값 = self.외적계산(스택[-2], 스택[-1], 현재점)
                print(f"      📏 외적({스택[-2]}, {스택[-1]}, {현재점}) = {외적값:.2f}")
                
                if 외적값 > 0:  # 반시계방향 (왼쪽 회전)
                    print(f"      ✅ 왼쪽 회전! {현재점} 추가")
                    break
                else:  # 시계방향 또는 직선
                    제거된점 = 스택.pop()
                    print(f"      ❌ 오른쪽 회전! {제거된점} 제거")
            
            스택.append(현재점)
            print(f"      📚 업데이트된 스택: {스택}")
        
        print(f"\n🎉 그라함 스캔 완료!")
        print(f"   🛡️ 볼록 껍질: {스택}")
        
        return 스택
    
    def 기프트_래핑(self, 점들):
        """기프트 래핑(Jarvis March) 알고리즘"""
        print(f"🌊 컨벡시: '기프트 래핑 마법으로도 보호막을 만들어보자!'")
        
        if len(점들) < 3:
            return 점들
        
        # 가장 왼쪽 점 찾기
        시작점 = min(점들, key=lambda p: p[0])
        print(f"   🎯 시작점: {시작점} (가장 왼쪽 점)")
        
        볼록껍질 = []
        현재점 = 시작점
        
        while True:
            볼록껍질.append(현재점)
            print(f"\n   🔍 현재점: {현재점}")
            
            # 다음 점을 찾기 위해 모든 점 확인
            다음점 = 점들[0]
            for 후보점 in 점들[1:]:
                외적값 = self.외적계산(현재점, 다음점, 후보점)
                print(f"      📏 외적({현재점}, {다음점}, {후보점}) = {외적값:.2f}")
                
                if 외적값 > 0:  # 후보점이 더 바깥쪽
                    다음점 = 후보점
                    print(f"      🔄 다음점 업데이트: {다음점}")
            
            print(f"   ➡️ 선택된 다음점: {다음점}")
            
            # 시작점으로 돌아오면 완료
            if 다음점 == 시작점:
                print(f"   🔄 시작점으로 돌아옴! 완료!")
                break
            
            현재점 = 다음점
        
        print(f"\n🎉 기프트 래핑 완료!")
        print(f"   🛡️ 볼록 껍질: {볼록껍질}")
        
        return 볼록껍질
    
    def 볼록껍질_넓이계산(self, 볼록껍질):
        """볼록 껍질의 넓이 계산"""
        if len(볼록껍질) < 3:
            return 0
        
        넓이 = 0
        n = len(볼록껍질)
        
        for i in range(n):
            j = (i + 1) % n
            넓이 += 볼록껍질[i][0] * 볼록껍질[j][1]
            넓이 -= 볼록껍질[j][0] * 볼록껍질[i][1]
        
        return abs(넓이) / 2
    
    def 시각화(self, 원본점들, 볼록껍질):
        """볼록 껍질 시각화"""
        print(f"🎨 '보호막을 예쁘게 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(10, 8))
            
            # 원본 점들
            x_coords = [p[0] for p in 원본점들]
            y_coords = [p[1] for p in 원본점들]
            plt.scatter(x_coords, y_coords, c='lightblue', s=100, alpha=0.7, label='모든 점들')
            
            # 점 번호 표시
            for i, (x, y) in enumerate(원본점들):
                plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
            
            # 볼록 껍질 그리기
            if len(볼록껍질) >= 3:
                껍질_x = [p[0] for p in 볼록껍질] + [볼록껍질[0][0]]  # 닫힌 다각형
                껍질_y = [p[1] for p in 볼록껍질] + [볼록껍질[0][1]]
                
                plt.plot(껍질_x, 껍질_y, 'r-', linewidth=2, label='볼록 껍질')
                plt.fill(껍질_x, 껍질_y, alpha=0.2, color='red', label='보호 영역')
                
                # 볼록 껍질 점들 강조
                plt.scatter([p[0] for p in 볼록껍질], [p[1] for p in 볼록껍질], 
                           c='red', s=150, zorder=5)
            
            넓이 = self.볼록껍질_넓이계산(볼록껍질)
            plt.title(f'🌊 컨벡시의 보호막 (넓이: {넓이:.2f})', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib가 필요해!'")

# 실전 예시
print("🌊 컨벡시의 보호막 생성 모험:")
컨벡시 = 컨벡시의_보호막생성()

# 테스트 점들
점들 = [(0, 3), (1, 1), (2, 2), (4, 4), (0, 0), (1, 2), (3, 1), (3, 3)]
print(f"\n🌟 요정 마을의 점들: {점들}")

print(f"\n" + "="*50)
print("📊 방법 1: 그라함 스캔")
그라함_결과 = 컨벡시.그라함_스캔(점들.copy())

print(f"\n" + "="*50)
print("📊 방법 2: 기프트 래핑") 
래핑_결과 = 컨벡시.기프트_래핑(점들.copy())

print(f"\n📋 결과 비교:")
print(f"   그라함 스캔: {그라함_결과}")
print(f"   기프트 래핑: {래핑_결과}")

그라함_넓이 = 컨벡시.볼록껍질_넓이계산(그라함_결과)
래핑_넓이 = 컨벡시.볼록껍질_넓이계산(래핑_결과)

print(f"   그라함 스캔 보호 면적: {그라함_넓이:.2f}")
print(f"   기프트 래핑 보호 면적: {래핑_넓이:.2f}")

# 시각화
컨벡시.시각화(점들, 그라함_결과)
```

**⏰ 시간복잡도**: 
- 그라함 스캔: O(n log n) - "정렬이 필요해!"
- 기프트 래핑: O(nh) - "h는 껍질 점의 개수!"

**💫 마법 속성**: 모든 점을 포함하는 최소 볼록 경계 생성

**🎯 장점**: 완벽한 보호막, 최소 면적
**⚠️ 단점**: 복잡한 계산, 3차원으로 확장시 어려움

---

## 🌟 제2막: 기하 마법사 길드의 등장

### ⚡ 인터섹 (Line Intersection) - "선분 교차의 예언자"

#### 캐릭터 설정 🔮
- **클래스**: 예언자 (Oracle)
- **정체**: 무수히 많은 선분들 사이의 교차점을 예측하고 찾아내는 미래시 요정
- **별명**: "교차점의 예언자"
- **성격**: 예리하고 통찰력이 뛰어남, 복잡한 패턴을 한눈에 파악
- **특기**: 스위프 라인과 기하학적 판정으로 모든 교차점 발견
- **철학**: "모든 만남에는 의미가 있고, 모든 교차에는 이유가 있다"
- **말버릇**: "선분들의 운명적 만남을 예언하겠어!"
- **무기**: 교차점 예언 수정구 (스위프 라인 알고리즘)

#### 선분 교차 탐지 모험 🌟
```python
import bisect
from collections import namedtuple

# 이벤트와 선분 정의
Event = namedtuple('Event', ['x', 'type', 'segment', 'point'])
Segment = namedtuple('Segment', ['id', 'start', 'end'])

class 인터섹의_교차예언:
    def __init__(self):
        print("⚡ 인터섹: '안녕! 선분들의 신비한 만남을 예언하는 인터섹이야!'")
    
    def 두선분_교차판정(self, 선분1, 선분2):
        """두 선분이 교차하는지 판정하고 교차점 찾기"""
        p1, q1 = 선분1.start, 선분1.end
        p2, q2 = 선분2.start, 선분2.end
        
        print(f"   🔍 선분 {선분1.id}: {p1} → {q1}")
        print(f"   🔍 선분 {선분2.id}: {p2} → {q2}")
        
        def 방향(p, q, r):
            """세 점의 방향 결정 (시계방향: 1, 반시계방향: -1, 직선: 0)"""
            외적 = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if 외적 > 0:
                return 1    # 시계방향
            elif 외적 < 0:
                return -1   # 반시계방향
            else:
                return 0    # 직선
        
        def 선분위의점(p, q, r):
            """점 r이 선분 pq 위에 있는지 확인"""
            return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                    min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))
        
        # 네 방향 계산
        d1 = 방향(p1, q1, p2)
        d2 = 방향(p1, q1, q2)
        d3 = 방향(p2, q2, p1)
        d4 = 방향(p2, q2, q1)
        
        print(f"      방향 판정: d1={d1}, d2={d2}, d3={d3}, d4={d4}")
        
        # 일반적인 교차 case
        if d1 != d2 and d3 != d4:
            # 교차점 계산
            교차점 = self.교차점_계산(선분1, 선분2)
            print(f"      ✅ 일반적 교차! 교차점: {교차점}")
            return True, 교차점
        
        # 특수한 경우들 (한 점이 다른 선분 위에 있는 경우)
        if d1 == 0 and 선분위의점(p1, q1, p2):
            print(f"      ✅ 특수 교차! 점 {p2}가 선분 {선분1.id} 위에 있음")
            return True, p2
        if d2 == 0 and 선분위의점(p1, q1, q2):
            print(f"      ✅ 특수 교차! 점 {q2}가 선분 {선분1.id} 위에 있음")
            return True, q2
        if d3 == 0 and 선분위의점(p2, q2, p1):
            print(f"      ✅ 특수 교차! 점 {p1}가 선분 {선분2.id} 위에 있음")
            return True, p1
        if d4 == 0 and 선분위의점(p2, q2, q1):
            print(f"      ✅ 특수 교차! 점 {q1}가 선분 {선분2.id} 위에 있음")
            return True, q1
        
        print(f"      ❌ 교차하지 않음")
        return False, None
    
    def 교차점_계산(self, 선분1, 선분2):
        """두 선분의 교차점 계산"""
        x1, y1 = 선분1.start
        x2, y2 = 선분1.end
        x3, y3 = 선분2.start
        x4, y4 = 선분2.end
        
        # 선분의 방향벡터와 교차점 계산
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # 평행선
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        교차점_x = x1 + t * (x2 - x1)
        교차점_y = y1 + t * (y2 - y1)
        
        return (round(교차점_x, 3), round(교차점_y, 3))
    
    def 무차별_교차탐지(self, 선분들):
        """무차별 대입으로 모든 교차점 찾기"""
        print(f"⚡ 인터섹: '무차별 대입으로 {len(선분들)}개 선분의 모든 교차점을 찾아보자!'")
        
        교차점들 = []
        비교횟수 = 0
        
        for i in range(len(선분들)):
            for j in range(i + 1, len(선분들)):
                비교횟수 += 1
                선분1 = 선분들[i]
                선분2 = 선분들[j]
                
                print(f"\n🔍 비교 {비교횟수}: 선분 {선분1.id} vs 선분 {선분2.id}")
                교차함, 교차점 = self.두선분_교차판정(선분1, 선분2)
                
                if 교차함:
                    교차점들.append((선분1.id, 선분2.id, 교차점))
                    print(f"   🌟 교차점 발견: {교차점}")
        
        print(f"\n🎉 무차별 탐지 완료!")
        print(f"   📊 총 비교 횟수: {비교횟수}")
        print(f"   ⭐ 발견된 교차점 수: {len(교차점들)}")
        
        return 교차점들
    
    def 스위프라인_교차탐지(self, 선분들):
        """스위프 라인 알고리즘으로 효율적 교차점 탐지"""
        print(f"⚡ 인터섹: '스위프 라인 마법으로 효율적으로 교차점을 찾아보자!'")
        
        # 이벤트 생성 (선분의 시작점과 끝점)
        이벤트들 = []
        for 선분 in 선분들:
            시작점 = min(선분.start, 선분.end, key=lambda p: p[0])  # x가 작은 점
            끝점 = max(선분.start, 선분.end, key=lambda p: p[0])    # x가 큰 점
            
            이벤트들.append(Event(시작점[0], 'start', 선분, 시작점))
            이벤트들.append(Event(끝점[0], 'end', 선분, 끝점))
        
        # x 좌표 순으로 이벤트 정렬
        이벤트들.sort(key=lambda e: (e.x, e.type == 'end'))  # 같은 x에서는 start가 먼저
        
        print(f"   📅 이벤트 순서:")
        for i, 이벤트 in enumerate(이벤트들):
            print(f"      {i+1}. x={이벤트.x}: {이벤트.type} 선분{이벤트.segment.id}")
        
        활성선분들 = []  # 현재 스위프 라인과 교차하는 선분들
        교차점들 = []
        
        for 이벤트 in 이벤트들:
            print(f"\n🔄 이벤트 처리: x={이벤트.x}, {이벤트.type} 선분{이벤트.segment.id}")
            
            if 이벤트.type == 'start':
                # 새로운 선분을 활성 목록에 추가
                활성선분들.append(이벤트.segment)
                print(f"   ➕ 선분 {이벤트.segment.id} 활성화")
                
                # 새 선분과 기존 활성 선분들 간의 교차 확인
                for 기존선분 in 활성선분들[:-1]:  # 방금 추가한 선분 제외
                    교차함, 교차점 = self.두선분_교차판정(이벤트.segment, 기존선분)
                    if 교차함 and 교차점[0] > 이벤트.x:  # 미래의 교차점만
                        교차점들.append((이벤트.segment.id, 기존선분.id, 교차점))
                        print(f"   🌟 미래 교차점 예측: {교차점}")
            
            elif 이벤트.type == 'end':
                # 선분을 활성 목록에서 제거
                if 이벤트.segment in 활성선분들:
                    활성선분들.remove(이벤트.segment)
                    print(f"   ➖ 선분 {이벤트.segment.id} 비활성화")
            
            print(f"   📋 현재 활성 선분들: {[s.id for s in 활성선분들]}")
        
        print(f"\n🎉 스위프 라인 완료!")
        print(f"   ⭐ 발견된 교차점 수: {len(교차점들)}")
        
        return 교차점들
    
    def 시각화(self, 선분들, 교차점들):
        """선분들과 교차점들 시각화"""
        print(f"🎨 '선분들의 운명적 만남을 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(12, 10))
            
            # 선분들 그리기
            색상들 = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, 선분 in enumerate(선분들):
                색상 = 색상들[i % len(색상들)]
                plt.plot([선분.start[0], 선분.end[0]], 
                        [선분.start[1], 선분.end[1]], 
                        color=색상, linewidth=2, label=f'선분 {선분.id}')
                
                # 선분 시작점과 끝점 표시
                plt.scatter(*선분.start, color=색상, s=100, zorder=5)
                plt.scatter(*선분.end, color=색상, s=100, zorder=5)
                
                # 선분 중점에 ID 표시
                중점_x = (선분.start[0] + 선분.end[0]) / 2
                중점_y = (선분.start[1] + 선분.end[1]) / 2
                plt.annotate(f'S{선분.id}', (중점_x, 중점_y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            
            # 교차점들 표시
            for i, (seg1_id, seg2_id, 점) in enumerate(교차점들):
                plt.scatter(*점, color='black', s=200, marker='*', 
                          zorder=10, label='교차점' if i == 0 else "")
                plt.annotate(f'P{i+1}', 점, xytext=(10, 10), 
                           textcoords='offset points', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.title('⚡ 인터섹의 선분 교차 예언', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib가 필요해!'")

# 실전 예시
print("⚡ 인터섹의 교차점 예언 모험:")
인터섹 = 인터섹의_교차예언()

# 테스트 선분들
선분들 = [
    Segment(1, (0, 0), (4, 4)),      # 대각선 1
    Segment(2, (0, 4), (4, 0)),      # 대각선 2  
    Segment(3, (1, 1), (3, 3)),      # 작은 대각선
    Segment(4, (2, 0), (2, 4)),      # 수직선
    Segment(5, (0, 2), (4, 2)),      # 수평선
]

print(f"\n🌟 요정 마을의 선분들:")
for 선분 in 선분들:
    print(f"   선분 {선분.id}: {선분.start} → {선분.end}")

print(f"\n" + "="*60)
print("📊 방법 1: 무차별 대입법")
무차별_교차점들 = 인터섹.무차별_교차탐지(선분들)

print(f"\n" + "="*60)
print("📊 방법 2: 스위프 라인 알고리즘")
스위프_교차점들 = 인터섹.스위프라인_교차탐지(선분들)

print(f"\n📋 결과 비교:")
print(f"   무차별 대입 교차점 수: {len(무차별_교차점들)}")
print(f"   스위프 라인 교차점 수: {len(스위프_교차점들)}")

print(f"\n🌟 발견된 교차점들:")
for i, (seg1, seg2, 점) in enumerate(무차별_교차점들):
    print(f"   교차점 {i+1}: 선분{seg1}과 선분{seg2}가 {점}에서 만남")

# 시각화
인터섹.시각화(선분들, 무차별_교차점들)
```

**⏰ 시간복잡도**: 
- 무차별 대입: O(n²) - "모든 선분 쌍을 확인!"
- 스위프 라인: O(n log n + k) - "k는 교차점 개수!"

**💫 마법 속성**: 선분들의 만남 예측과 교차점 발견

**🎯 장점**: 모든 교차점 정확히 탐지
**⚠️ 단점**: 복잡한 구현, 특수 케이스 처리 필요

---

### 📐 트라이앵 (Delaunay Triangulation) - "삼각분할의 건축가"

#### 캐릭터 설정 🏗️
- **클래스**: 건축가 (Architect)
- **정체**: 점들을 가장 아름다운 삼각형으로 연결하는 기하학적 예술가
- **별명**: "황금 삼각형의 건축가"
- **성격**: 완벽주의적이고 예술적, 아름다운 기하학적 구조를 추구
- **특기**: 들로네 조건을 만족하는 최적의 삼각분할 생성
- **철학**: "가장 균등하고 아름다운 삼각형이 가장 좋은 분할이다"
- **말버릇**: "완벽한 삼각형을 만들어보자!"
- **무기**: 황금 컴패스 (최적 삼각분할 도구)

#### 들로네 삼각분할 모험 🔺
```python
import math
import numpy as np
from collections import defaultdict

class 트라이앵의_삼각건축:
    def __init__(self):
        print("📐 트라이앵: '안녕! 아름다운 삼각분할을 만드는 건축가 트라이앵이야!'")
    
    def 외심_계산(self, p1, p2, p3):
        """삼각형의 외심과 외접원 반지름 계산"""
        x1, y1 = p1
        x2, y2 = p2  
        x3, y3 = p3
        
        # 외심 계산 공식
        D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        
        if abs(D) < 1e-10:  # 거의 일직선상의 점들
            return None, float('inf')
        
        ux = ((x1*x1 + y1*y1) * (y2 - y3) + 
              (x2*x2 + y2*y2) * (y3 - y1) + 
              (x3*x3 + y3*y3) * (y1 - y2)) / D
        
        uy = ((x1*x1 + y1*y1) * (x3 - x2) + 
              (x2*x2 + y2*y2) * (x1 - x3) + 
              (x3*x3 + y3*y3) * (x2 - x1)) / D
        
        외심 = (ux, uy)
        반지름 = math.sqrt((ux - x1)**2 + (uy - y1)**2)
        
        return 외심, 반지름
    
    def 외접원_내부_확인(self, p1, p2, p3, test_point):
        """점이 삼각형의 외접원 내부에 있는지 확인"""
        외심, 반지름 = self.외심_계산(p1, p2, p3)
        
        if 외심 is None:
            return False
        
        거리 = math.sqrt((test_point[0] - 외심[0])**2 + (test_point[1] - 외심[1])**2)
        return 거리 < 반지름 - 1e-10  # 수치 오차 고려
    
    def 단순_들로네_삼각분할(self, 점들):
        """단순한 들로네 삼각분할 (작은 점 집합용)"""
        print(f"📐 트라이앵: '단순 알고리즘으로 {len(점들)}개 점의 아름다운 삼각분할을 만들어보자!'")
        
        if len(점들) < 3:
            print("😅 '점이 3개 미만이면 삼각형을 만들 수 없어!'")
            return []
        
        # 모든 가능한 삼각형 생성
        가능한_삼각형들 = []
        n = len(점들)
        
        print(f"   🔍 모든 가능한 삼각형 조합 확인:")
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = 점들[i], 점들[j], 점들[k]
                    
                    # 일직선상이 아닌지 확인
                    외심, 반지름 = self.외심_계산(p1, p2, p3)
                    if 외심 is None:
                        continue
                    
                    print(f"      🔺 삼각형 ({i+1},{j+1},{k+1}): {p1}, {p2}, {p3}")
                    print(f"         외심: {외심}, 반지름: {반지름:.3f}")
                    
                    # 들로네 조건 확인: 외접원 내부에 다른 점이 없어야 함
                    들로네_조건_만족 = True
                    
                    for l in range(n):
                        if l == i or l == j or l == k:
                            continue
                        
                        if self.외접원_내부_확인(p1, p2, p3, 점들[l]):
                            print(f"         ❌ 점 {l+1}{점들[l]}이 외접원 내부에 있음!")
                            들로네_조건_만족 = False
                            break
                    
                    if 들로네_조건_만족:
                        가능한_삼각형들.append((i, j, k))
                        print(f"         ✅ 들로네 조건 만족! 삼각형 추가")
                    else:
                        print(f"         ❌ 들로네 조건 불만족")
        
        print(f"\n🎉 들로네 삼각분할 완료!")
        print(f"   🔺 생성된 삼각형 수: {len(가능한_삼각형들)}")
        
        return 가능한_삼각형들
    
    def 점들의_볼록껍질_확인(self, 점들):
        """점들이 볼록 위치에 있는지 확인 (단순화된 버전)"""
        if len(점들) <= 3:
            return True
        
        # 단순히 모든 점이 다른 세 점으로 이루어진 삼각형 내부에 있지 않은지 확인
        n = len(점들)
        for i in range(n):
            내부에_있음 = False
            for j in range(n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if i == j or i == k or i == l:
                            continue
                        if self.점이_삼각형_내부에_있는가(점들[i], 점들[j], 점들[k], 점들[l]):
                            내부에_있음 = True
                            break
                if 내부에_있음:
                    break
            if not 내부에_있음:
                return False
        return True
    
    def 점이_삼각형_내부에_있는가(self, p, a, b, c):
        """점 p가 삼각형 abc 내부에 있는지 확인"""
        def 부호(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = 부호(p, a, b)
        d2 = 부호(p, b, c)
        d3 = 부호(p, c, a)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def 삼각분할_품질_평가(self, 점들, 삼각형들):
        """삼각분할의 품질 평가 (각도 분포)"""
        print(f"\n📊 트라이앵: '만든 삼각분할의 품질을 평가해보자!'")
        
        모든_각도들 = []
        
        for i, (idx1, idx2, idx3) in enumerate(삼각형들):
            p1, p2, p3 = 점들[idx1], 점들[idx2], 점들[idx3]
            
            # 각 꼭짓점에서의 각도 계산
            def 각도_계산(center, point1, point2):
                v1 = (point1[0] - center[0], point1[1] - center[1])
                v2 = (point2[0] - center[0], point2[1] - center[1])
                
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
                magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if magnitude1 * magnitude2 == 0:
                    return 0
                
                cos_angle = dot_product / (magnitude1 * magnitude2)
                cos_angle = max(-1, min(1, cos_angle))  # clamp to [-1, 1]
                
                return math.degrees(math.acos(cos_angle))
            
            각1 = 각도_계산(p1, p2, p3)
            각2 = 각도_계산(p2, p3, p1)
            각3 = 각도_계산(p3, p1, p2)
            
            모든_각도들.extend([각1, 각2, 각3])
            
            print(f"   🔺 삼각형 {i+1}: 각도들 = {각1:.1f}°, {각2:.1f}°, {각3:.1f}°")
        
        최소각 = min(모든_각도들)
        최대각 = max(모든_각도들)
        평균각 = sum(모든_각도들) / len(모든_각도들)
        
        print(f"\n📈 삼각분할 품질 분석:")
        print(f"   📐 최소 각도: {최소각:.1f}° (클수록 좋음)")
        print(f"   📐 최대 각도: {최대각:.1f}° (작을수록 좋음)")
        print(f"   📐 평균 각도: {평균각:.1f}° (60°에 가까울수록 좋음)")
        
        return 최소각, 최대각, 평균각
    
    def 시각화(self, 점들, 삼각형들):
        """들로네 삼각분할 시각화"""
        print(f"🎨 '아름다운 삼각형들을 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import random
            
            plt.figure(figsize=(12, 10))
            
            # 점들 그리기
            x_coords = [p[0] for p in 점들]
            y_coords = [p[1] for p in 점들]
            plt.scatter(x_coords, y_coords, c='red', s=100, zorder=5)
            
            # 점 번호 표시
            for i, (x, y) in enumerate(점들):
                plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=12, fontweight='bold')
            
            # 삼각형들 그리기
            색상들 = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                     'lightpink', 'lightcyan', 'wheat', 'plum']
            
            for i, (idx1, idx2, idx3) in enumerate(삼각형들):
                p1, p2, p3 = 점들[idx1], 점들[idx2], 점들[idx3]
                
                # 삼각형 채우기
                삼각형 = patches.Polygon([p1, p2, p3], 
                                      facecolor=색상들[i % len(색상들)], 
                                      alpha=0.3, edgecolor='black', linewidth=1)
                plt.gca().add_patch(삼각형)
                
                # 삼각형 중심에 번호 표시
                중심_x = (p1[0] + p2[0] + p3[0]) / 3
                중심_y = (p1[1] + p2[1] + p3[1]) / 3
                plt.annotate(f'T{i+1}', (중심_x, 중심_y), 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                
                # 외접원 그리기 (선택적)
                외심, 반지름 = self.외심_계산(p1, p2, p3)
                if 외심 is not None:
                    원 = patches.Circle(외심, 반지름, fill=False, 
                                      color='gray', linestyle='--', alpha=0.5)
                    plt.gca().add_patch(원)
                    plt.scatter(*외심, c='gray', s=30, marker='+', alpha=0.7)
            
            plt.title('📐 트라이앵의 들로네 삼각분할', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib가 필요해!'")

# 실전 예시
print("📐 트라이앵의 삼각분할 건축 모험:")
트라이앵 = 트라이앵의_삼각건축()

# 테스트 점들 (정사각형 형태로 배치)
점들 = [(0, 0), (3, 0), (3, 3), (0, 3), (1.5, 1.5)]  # 중앙에 점 하나 추가

print(f"\n🌟 요정 마을의 점들:")
for i, 점 in enumerate(점들):
    print(f"   점 {i+1}: {점}")

print(f"\n" + "="*60)
삼각형들 = 트라이앵.단순_들로네_삼각분할(점들)

# 품질 평가
트라이앵.삼각분할_품질_평가(점들, 삼각형들)

# 시각화
트라이앵.시각화(점들, 삼각형들)
```

**⏰ 시간복잡도**: 단순 알고리즘 O(n⁴), 고급 알고리즘 O(n log n)
**💫 마법 속성**: 최적의 삼각형 배치로 아름다운 기하 구조 생성
**🎯 장점**: 균등한 삼각형, 수치 해석에 적합
**⚠️ 단점**: 복잡한 구현, 정밀한 수치 계산 필요

---

## 🌟 제3막: 영토 관리소의 마법

### 🗺️ 보로노이 (Voronoi Diagram) - "영역 분할의 영주"

#### 캐릭터 설정 👑
- **클래스**: 영주 (Lord)
- **정체**: 각 점의 영향 범위를 공정하게 나누어 관리하는 영토 관리자
- **별명**: "공정한 영토 분배자"
- **성격**: 공정하고 체계적, 모든 영역을 균등하게 관리하고자 함
- **특기**: 보로노이 셀 생성으로 최적의 영토 분할
- **철학**: "각자의 영역은 공정하게, 경계는 명확하게"
- **말버릿**: "공정한 영토 분할을 시작하자!"
- **무기**: 영역 분할 나침반 (거리 기반 경계 계산)

#### 보로노이 다이어그램 생성 모험 🏰
```python
import math
import numpy as np
from collections import defaultdict

class 보로노이의_영토분할:
    def __init__(self):
        print("🗺️ 보로노이: '안녕! 공정한 영토 분할을 담당하는 영주 보로노이야!'")
    
    def 거리계산(self, p1, p2):
        """두 점 사이의 유클리드 거리"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def 단순_보로노이_생성(self, 시드점들, 경계영역):
        """단순한 보로노이 다이어그램 생성 (격자 샘플링 방식)"""
        print(f"🗺️ 보로노이: '{len(시드점들)}개 시드점으로 영토를 분할하겠다!'")
        
        min_x, max_x, min_y, max_y = 경계영역
        해상도 = 50  # 격자 해상도
        
        # 격자점들 생성
        x_vals = np.linspace(min_x, max_x, 해상도)
        y_vals = np.linspace(min_y, max_y, 해상도)
        
        영토맵 = {}  # (x, y) -> 가장 가까운 시드점 인덱스
        
        print(f"   📊 {해상도}x{해상도} 격자로 영토 분할 진행")
        
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                격자점 = (x, y)
                
                # 가장 가까운 시드점 찾기
                최소거리 = float('inf')
                가장가까운_시드 = 0
                
                for k, 시드점 in enumerate(시드점들):
                    거리 = self.거리계산(격자점, 시드점)
                    if 거리 < 최소거리:
                        최소거리 = 거리  
                        가장가까운_시드 = k
                
                영토맵[(x, y)] = 가장가까운_시드
        
        print(f"   🎉 영토 분할 완료! 각 격자점이 소속 영주를 결정했어!")
        
        return 영토맵, (x_vals, y_vals)
    
    def 경계선_추출(self, 영토맵, 격자정보):
        """보로노이 셀들 간의 경계선 추출"""
        print(f"   🔍 영토 간 경계선을 찾아보자!")
        
        x_vals, y_vals = 격자정보
        경계점들 = []
        
        for i in range(len(x_vals) - 1):
            for j in range(len(y_vals) - 1):
                현재점 = (x_vals[i], y_vals[j])
                오른쪽점 = (x_vals[i+1], y_vals[j])
                아래점 = (x_vals[i], y_vals[j+1])
                
                # 인접한 점들이 다른 영토에 속하면 경계
                if (현재점 in 영토맵 and 오른쪽점 in 영토맵 and
                    영토맵[현재점] != 영토맵[오른쪽점]):
                    경계점들.append((현재점, 오른쪽점))
                
                if (현재점 in 영토맵 and 아래점 in 영토맵 and
                    영토맵[현재점] != 영토맵[아래점]):
                    경계점들.append((현재점, 아래점))
        
        print(f"   📏 총 {len(경계점들)}개의 경계선 발견!")
        return 경계점들
    
    def 영토_통계_분석(self, 시드점들, 영토맵):
        """각 영토의 통계 분석"""
        print(f"\n📊 보로노이: '각 영주의 영토 현황을 분석해보자!'")
        
        영토별_격자수 = defaultdict(int)
        
        for 격자점, 소속영주 in 영토맵.items():
            영토별_격자수[소속영주] += 1
        
        전체_격자수 = len(영토맵)
        
        for i, 시드점 in enumerate(시드점들):
            격자수 = 영토별_격자수[i]
            비율 = (격자수 / 전체_격자수) * 100
            
            print(f"   👑 영주 {i+1} (시드점 {시드점}):")
            print(f"      📊 관할 격자: {격자수}개 ({비율:.1f}%)")
        
        # 영토 분배의 공정성 평가
        평균_격자수 = 전체_격자수 / len(시드점들)
        분산 = sum((영토별_격자수[i] - 평균_격자수)**2 for i in range(len(시드점들))) / len(시드점들)
        표준편차 = math.sqrt(분산)
        
        print(f"\n⚖️ 공정성 분석:")
        print(f"   📊 평균 영토 크기: {평균_격자수:.1f}격자")
        print(f"   📊 표준편차: {표준편차:.1f} (작을수록 공정)")
        
        return 영토별_격자수
    
    def 보로노이_응용_예시(self, 시드점들):
        """보로노이 다이어그램의 실제 응용 예시"""
        print(f"\n🌟 보로노이: '실제 세상에서 이렇게 활용돼!'")
        
        응용예시들 = [
            "🏪 상권 분석: 각 상점의 영향권 분석",
            "🏥 의료시설: 가장 가까운 병원 찾기",
            "📡 기지국 배치: 통신 커버리지 최적화",
            "🌧️ 기상관측: 강수량 분포 예측",
            "🗳️ 선거구 획정: 공정한 구역 분할",
            "🎮 게임 AI: 영역 제어 전략",
            "🧬 생물학: 세포 성장 패턴 모델링"
        ]
        
        for i, 예시 in enumerate(응용예시들):
            if i < len(시드점들):
                print(f"   {예시}")
    
    def 시각화(self, 시드점들, 영토맵, 격자정보, 경계선들):
        """보로노이 다이어그램 시각화"""
        print(f"🎨 '아름다운 영토 분할을 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(12, 10))
            
            x_vals, y_vals = 격자정보
            
            # 영토맵을 2D 배열로 변환
            영토_배열 = np.zeros((len(y_vals), len(x_vals)))
            
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    if (x, y) in 영토맵:
                        영토_배열[j, i] = 영토맵[(x, y)]
            
            # 영토 영역 색칠
            색상맵 = plt.cm.Set3  # 부드러운 색상 팔레트
            im = plt.imshow(영토_배열, extent=[min(x_vals), max(x_vals), min(y_vals), max(y_vals)],
                           cmap=색상맵, alpha=0.6, origin='lower')
            
            # 시드점들 표시
            for i, (x, y) in enumerate(시드점들):
                plt.scatter(x, y, c='red', s=200, marker='*', 
                          edgecolors='black', linewidth=2, zorder=5)
                plt.annotate(f'영주{i+1}', (x, y), xytext=(10, 10), 
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            # 경계선 그리기
            for 시작점, 끝점 in 경계선들:
                plt.plot([시작점[0], 끝점[0]], [시작점[1], 끝점[1]], 
                        'black', linewidth=1, alpha=0.8)
            
            plt.title('🗺️ 보로노이의 공정한 영토 분할', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            
            # 컬러바 추가
            cbar = plt.colorbar(im, ticks=range(len(시드점들)))
            cbar.set_label('영주 번호', rotation=270, labelpad=20)
            cbar.set_ticklabels([f'영주{i+1}' for i in range(len(시드점들))])
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib와 numpy가 필요해!'")

# 실전 예시
print("🗺️ 보로노이의 영토 분할 모험:")
보로노이 = 보로노이의_영토분할()

# 시드점들 (각 영주의 성 위치)
시드점들 = [(2, 2), (7, 8), (12, 3), (5, 7), (10, 10)]
경계영역 = (0, 15, 0, 12)  # (min_x, max_x, min_y, max_y)

print(f"\n🏰 영주들의 성 위치:")
for i, 성위치 in enumerate(시드점들):
    print(f"   영주 {i+1}의 성: {성위치}")

print(f"\n" + "="*60)
print("🗺️ 영토 분할 시작!")

# 보로노이 다이어그램 생성
영토맵, 격자정보 = 보로노이.단순_보로노이_생성(시드점들, 경계영역)

# 경계선 추출
경계선들 = 보로노이.경계선_추출(영토맵, 격자정보)

# 통계 분석
영토_통계 = 보로노이.영토_통계_분석(시드점들, 영토맵)

# 응용 예시 설명
보로노이.보로노이_응용_예시(시드점들)

# 시각화
보로노이.시각화(시드점들, 영토맵, 격자정보, 경계선들)
```

**⏰ 시간복잡도**: 단순 방식 O(n×m), 고급 방식 O(n log n)
**💫 마법 속성**: 공정한 영역 분할과 최근접 영역 결정
**🎯 장점**: 직관적이고 공정한 분할, 다양한 응용 가능
**⚠️ 단점**: 정확한 경계 계산 복잡, 고차원 확장 어려움

---

### 🎯 폴리고니 (Point in Polygon) - "경계 판별의 마법사"

#### 캐릭터 설정 🎭
- **클래스**: 판별사 (Judge)
- **정체**: 주어진 점이 다각형 내부에 있는지 정확히 판단하는 경계 전문가
- **별명**: "경계의 심판관"
- **성격**: 정확하고 논리적, 경계에 대한 명확한 판단을 내림
- **특기**: 레이 캐스팅과 회전각 계산으로 내외부 판정
- **철학**: "경계가 명확해야 세상이 질서정연하다"
- **말버릇**: "경계 안인가, 밖인가, 그것이 문제로다!"
- **무기**: 경계 판정 지팡이 (레이 캐스팅)

#### 점-다각형 내부 판정 모험 🔍
```python
import math

class 폴리고니의_경계판정:
    def __init__(self):
        print("🎯 폴리고니: '안녕! 경계의 심판관 폴리고니야! 점이 어디에 있는지 정확히 알려줄게!'")
    
    def 레이캐스팅_판정(self, 점, 다각형):
        """레이 캐스팅 알고리즘으로 점-다각형 내부 판정"""
        print(f"🎯 폴리고니: '레이 캐스팅으로 점 {점}이 다각형 내부에 있는지 판정해보자!'")
        
        x, y = 점
        n = len(다각형)
        교차횟수 = 0
        
        print(f"   🏹 점 {점}에서 오른쪽으로 수평선을 그어보자!")
        print(f"   📐 다각형 꼭짓점: {다각형}")
        
        for i in range(n):
            # 현재 변: 다각형[i] -> 다각형[(i+1)%n]
            x1, y1 = 다각형[i]
            x2, y2 = 다각형[(i + 1) % n]
            
            print(f"\n   🔍 변 {i+1}: ({x1}, {y1}) → ({x2}, {y2})")
            
            # 수평선과 교차 가능성 확인
            if min(y1, y2) < y <= max(y1, y2):
                print(f"      📏 y좌표 범위 [{min(y1, y2)}, {max(y1, y2)}]에 점의 y={y}가 포함됨")
                
                # 교차점의 x좌표 계산
                if y1 != y2:  # 수평선이 아닌 경우
                    교차점_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    print(f"      🎯 교차점 x좌표: {교차점_x:.3f}")
                    
                    if 교차점_x > x:  # 점의 오른쪽에서 교차
                        교차횟수 += 1
                        print(f"      ✅ 점의 오른쪽에서 교차! (교차횟수: {교차횟수})")
                    else:
                        print(f"      ❌ 점의 왼쪽에서 교차 (카운트 안함)")
                else:
                    print(f"      ➡️ 수평선이므로 무시")
            else:
                print(f"      ❌ y좌표 범위 밖 (교차 없음)")
        
        내부여부 = 교차횟수 % 2 == 1
        
        print(f"\n📊 최종 결과:")
        print(f"   🏹 총 교차 횟수: {교차횟수}")
        print(f"   🎯 판정: {'내부' if 내부여부 else '외부'} ({'홀수' if 내부여부 else '짝수'} 규칙)")
        
        return 내부여부
    
    def 회전각_판정(self, 점, 다각형):
        """회전각 합계 방법으로 점-다각형 내부 판정"""
        print(f"🎯 폴리고니: '회전각 방법으로도 점 {점}을 판정해보자!'")
        
        x, y = 점
        n = len(다각형)
        총_회전각 = 0.0
        
        print(f"   🔄 점에서 각 꼭짓점으로의 각도를 계산해보자!")
        
        for i in range(n):
            # 현재 꼭짓점과 다음 꼭짓점
            x1, y1 = 다각형[i]
            x2, y2 = 다각형[(i + 1) % n]
            
            # 점에서 각 꼭짓점으로의 벡터
            dx1, dy1 = x1 - x, y1 - y
            dx2, dy2 = x2 - x, y2 - y
            
            # 각도 계산
            각도1 = math.atan2(dy1, dx1)
            각도2 = math.atan2(dy2, dx2)
            
            # 각도 차이 계산 (-π ~ π 범위로 정규화)
            각도차 = 각도2 - 각도1
            
            # -π ~ π 범위로 조정
            while 각도차 > math.pi:
                각도차 -= 2 * math.pi
            while 각도차 < -math.pi:
                각도차 += 2 * math.pi
            
            총_회전각 += 각도차
            
            print(f"   📐 변 {i+1}: 각도 {math.degrees(각도1):.1f}° → {math.degrees(각도2):.1f}° (차이: {math.degrees(각도차):.1f}°)")
        
        총_회전각_도 = math.degrees(총_회전각)
        내부여부 = abs(총_회전각) > math.pi  # 총 회전각이 ±2π에 가까우면 내부
        
        print(f"\n📊 회전각 분석:")
        print(f"   🔄 총 회전각: {총_회전각_도:.1f}°")
        print(f"   🎯 판정: {'내부' if 내부여부 else '외부'} ({'±360°' if 내부여부 else '0°'} 규칙)")
        
        return 내부여부
    
    def 경계_위_점_확인(self, 점, 다각형, 허용오차=1e-10):
        """점이 다각형의 경계(변) 위에 있는지 확인"""
        print(f"   🔍 점 {점}이 경계선 위에 있는지 확인해보자!")
        
        x, y = 점
        n = len(다각형)
        
        for i in range(n):
            x1, y1 = 다각형[i]
            x2, y2 = 다각형[(i + 1) % n]
            
            # 점이 선분 위에 있는지 확인
            # 1. 점이 직선 위에 있는가?
            외적 = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
            
            if abs(외적) < 허용오차:  # 직선 위에 있음
                # 2. 점이 선분 범위 내에 있는가?
                if (min(x1, x2) <= x <= max(x1, x2) and 
                    min(y1, y2) <= y <= max(y1, y2)):
                    print(f"      ✅ 점이 변 {i+1} ({x1}, {y1}) → ({x2}, {y2}) 위에 있음!")
                    return True, i + 1
        
        print(f"      ❌ 점이 어떤 경계선 위에도 있지 않음")
        return False, None
    
    def 종합_판정(self, 점, 다각형):
        """모든 방법을 종합한 최종 판정"""
        print(f"🎯 폴리고니: '점 {점}에 대한 종합 판정을 시작하겠다!'")
        
        # 1. 경계 위 점 확인
        경계위_여부, 변번호 = self.경계_위_점_확인(점, 다각형)
        
        if 경계위_여부:
            print(f"🎯 최종 판정: 점은 다각형의 변 {변번호} 위에 있음 (경계)")
            return "경계"
        
        # 2. 레이 캐스팅 판정
        레이캐스팅_결과 = self.레이캐스팅_판정(점, 다각형)
        
        # 3. 회전각 판정 (검증용)
        회전각_결과 = self.회전각_판정(점, 다각형)
        
        # 결과 비교
        일치여부 = 레이캐스팅_결과 == 회전각_결과
        
        print(f"\n🎯 최종 판정 결과:")
        print(f"   🏹 레이 캐스팅: {'내부' if 레이캐스팅_결과 else '외부'}")
        print(f"   🔄 회전각 방법: {'내부' if 회전각_결과 else '외부'}")
        print(f"   ✅ 결과 일치: {'예' if 일치여부 else '아니오'}")
        
        if 일치여부:
            return "내부" if 레이캐스팅_결과 else "외부"
        else:
            print(f"   ⚠️ 경고: 두 방법의 결과가 다릅니다! 추가 검토 필요")
            return "불확실"
    
    def 시각화(self, 다각형, 테스트점들, 판정결과들):
        """점-다각형 판정 결과 시각화"""
        print(f"🎨 '경계 판정 결과를 예쁘게 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            plt.figure(figsize=(12, 10))
            
            # 다각형 그리기
            다각형_패치 = patches.Polygon(다각형, facecolor='lightblue', 
                                     alpha=0.3, edgecolor='black', linewidth=2)
            plt.gca().add_patch(다각형_패치)
            
            # 다각형 꼭짓점 표시
            for i, (x, y) in enumerate(다각형):
                plt.scatter(x, y, c='blue', s=100, zorder=5)
                plt.annotate(f'V{i+1}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
            
            # 다각형 변 번호 표시
            for i in range(len(다각형)):
                x1, y1 = 다각형[i]
                x2, y2 = 다각형[(i + 1) % len(다각형)]
                중점_x, 중점_y = (x1 + x2) / 2, (y1 + y2) / 2
                plt.annotate(f'E{i+1}', (중점_x, 중점_y), 
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", alpha=0.8))
            
            # 테스트 점들과 판정 결과 표시
            색상_매핑 = {'내부': 'red', '외부': 'green', '경계': 'orange', '불확실': 'purple'}
            기호_매핑 = {'내부': 'o', '외부': 's', '경계': '^', '불확실': 'X'}
            
            for i, (점, 판정) in enumerate(zip(테스트점들, 판정결과들)):
                색상 = 색상_매핑.get(판정, 'gray')
                기호 = 기호_매핑.get(판정, 'o')
                
                plt.scatter(*점, c=색상, s=150, marker=기호, 
                          edgecolors='black', linewidth=1, zorder=6)
                plt.annotate(f'P{i+1}({판정})', 점, xytext=(10, 10), 
                           textcoords='offset points', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=색상, alpha=0.7))
            
            # 범례 추가
            from matplotlib.lines import Line2D
            범례_요소들 = []
            for 판정, 색상 in 색상_매핑.items():
                if 판정 in 판정결과들:
                    기호 = 기호_매핑[판정]
                    범례_요소들.append(Line2D([0], [0], marker=기호, color='w', 
                                         markerfacecolor=색상, markersize=10, 
                                         markeredgecolor='black', label=판정))
            
            plt.legend(handles=범례_요소들, loc='upper right')
            plt.title('🎯 폴리고니의 경계 판정', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib가 필요해!'")

# 실전 예시
print("🎯 폴리고니의 경계 판정 모험:")
폴리고니 = 폴리고니의_경계판정()

# 테스트용 다각형 (집 모양)
다각형 = [(1, 1), (4, 1), (4, 3), (2.5, 4), (1, 3)]

# 테스트할 점들
테스트점들 = [
    (2.5, 2),    # 내부
    (0, 0),      # 외부
    (2.5, 1),    # 경계 (아래변)
    (5, 2),      # 외부
    (2.5, 3.5),  # 내부 (지붕 부분)
    (1, 2),      # 경계 (왼쪽변)
    (3, 0.5)     # 외부
]

print(f"\n🏠 테스트 다각형 (집 모양): {다각형}")
print(f"🎯 테스트할 점들: {테스트점들}")

판정결과들 = []

for i, 점 in enumerate(테스트점들):
    print(f"\n" + "="*60)
    print(f"📍 테스트 {i+1}: 점 {점}")
    
    판정결과 = 폴리고니.종합_판정(점, 다각형)
    판정결과들.append(판정결과)
    
    print(f"🎯 폴리고니: '점 {점}은 다각형의 {판정결과}에 있다!'")

print(f"\n" + "="*60)
print("📋 전체 결과 요약:")
for i, (점, 판정) in enumerate(zip(테스트점들, 판정결과들)):
    상태표시 = "✅" if 판정 != "불확실" else "⚠️"
    print(f"   {상태표시} 점 {i+1} {점}: {판정}")

# 시각화
폴리고니.시각화(다각형, 테스트점들, 판정결과들)
```

**⏰ 시간복잡도**: O(n) - "n은 다각형의 꼭짓점 수"
**💫 마법 속성**: 정확한 내외부 판정과 경계 감지
**🎯 장점**: 정확하고 신뢰할 수 있는 판정
**⚠️ 단점**: 복잡한 다각형에서 특수 케이스 처리 필요

---

## 🌟 제4막: 변환 마법 학원의 등장

### 🔄 로테이터 (Rotation Transform) - "회전 춤의 요정"

#### 캐릭터 설정 💃
- **클래스**: 춤꾼 (Dancer)
- **정체**: 점들을 아름답게 회전시키는 변환의 무용수
- **별명**: "회전의 여신"
- **성격**: 우아하고 예술적, 모든 움직임이 춤처럼 아름다움
- **특기**: 임의의 중심점과 각도로 정확한 회전 변환
- **철학**: "모든 변화는 아름다운 춤이다"
- **말버릿**: "함께 춤을 춰보자!"
- **무기**: 회전 무도선 (변환 매트릭스)

#### 회전 변환 모험 💫
```python
import math

class 로테이터의_회전춤:
    def __init__(self):
        print("🔄 로테이터: '안녕! 아름다운 회전 춤을 추는 로테이터야!'")
    
    def 단순_회전(self, 점, 각도_도):
        """원점을 중심으로 한 단순 회전"""
        print(f"🔄 로테이터: '점 {점}을 원점 중심으로 {각도_도}도 회전시켜보자!'")
        
        x, y = 점
        각도_라디안 = math.radians(각도_도)
        
        print(f"   📐 회전 각도: {각도_도}° = {각도_라디안:.4f} 라디안")
        print(f"   🎭 변환 매트릭스:")
        print(f"      [cos({각도_도}°)  -sin({각도_도}°)]   [{math.cos(각도_라디안):.3f}  {-math.sin(각도_라디안):.3f}]")
        print(f"      [sin({각도_도}°)   cos({각도_도}°)]   [{math.sin(각도_라디안):.3f}   {math.cos(각도_라디안):.3f}]")
        
        # 회전 변환 공식: x' = x*cos(θ) - y*sin(θ), y' = x*sin(θ) + y*cos(θ)
        새_x = x * math.cos(각도_라디안) - y * math.sin(각도_라디안)
        새_y = x * math.sin(각도_라디안) + y * math.cos(각도_라디안)
        
        새점 = (round(새_x, 3), round(새_y, 3))
        
        print(f"   ✨ 변환 계산:")
        print(f"      x' = {x} × {math.cos(각도_라디안):.3f} - {y} × {math.sin(각도_라디안):.3f} = {새_x:.3f}")
        print(f"      y' = {x} × {math.sin(각도_라디안):.3f} + {y} × {math.cos(각도_라디안):.3f} = {새_y:.3f}")
        print(f"   🎉 회전 결과: {점} → {새점}")
        
        return 새점
    
    def 중심점_회전(self, 점, 중심점, 각도_도):
        """임의의 중심점을 기준으로 한 회전"""
        print(f"🔄 로테이터: '점 {점}을 중심점 {중심점} 기준으로 {각도_도}도 회전시켜보자!'")
        
        x, y = 점
        cx, cy = 중심점
        
        print(f"   📍 1단계: 중심점을 원점으로 평행이동")
        평행이동된점 = (x - cx, y - cy)
        print(f"      {점} - {중심점} = {평행이동된점}")
        
        print(f"   🔄 2단계: 원점 중심 회전")
        회전된점 = self.단순_회전(평행이동된점, 각도_도)
        
        print(f"   📍 3단계: 중심점만큼 다시 평행이동")
        최종점 = (회전된점[0] + cx, 회전된점[1] + cy)
        print(f"      {회전된점} + {중심점} = {최종점}")
        
        print(f"   🎉 최종 결과: {점} → {최종점}")
        
        return 최종점
    
    def 다중점_회전(self, 점들, 중심점, 각도_도):
        """여러 점들을 동시에 회전"""
        print(f"🔄 로테이터: '{len(점들)}개 점들을 함께 회전 춤추게 해보자!'")
        
        회전된점들 = []
        
        for i, 점 in enumerate(점들):
            print(f"\n   💃 점 {i+1} 회전:")
            회전된점 = self.중심점_회전(점, 중심점, 각도_도)
            회전된점들.append(회전된점)
        
        print(f"\n🎭 전체 회전 춤 완료!")
        print(f"   원본: {점들}")
        print(f"   변환: {회전된점들}")
        
        return 회전된점들
    
    def 회전_방향_설명(self, 각도_도):
        """회전 방향과 의미 설명"""
        print(f"\n🧭 로테이터: '회전 방향을 설명해줄게!'")
        
        if 각도_도 > 0:
            방향 = "반시계방향 (양의 각도)"
            의미 = "수학적 양의 방향"
        elif 각도_도 < 0:
            방향 = "시계방향 (음의 각도)"
            의미 = "시계 바늘 방향"
        else:
            방향 = "회전 없음"
            의미 = "원래 위치 유지"
        
        print(f"   🔄 {각도_도}도 회전: {방향}")
        print(f"   📚 의미: {의미}")
        
        # 특수 각도 설명
        특수각도들 = {
            90: "1/4 회전 (직각)",
            180: "1/2 회전 (반바퀴)",
            270: "3/4 회전",
            360: "1 회전 (한바퀴, 원래 위치)",
            -90: "시계방향 1/4 회전",
            -180: "시계방향 1/2 회전"
        }
        
        if abs(각도_도) in 특수각도들:
            print(f"   🌟 특수각: {특수각도들[abs(각도_도)]}")
    
    def 시각화(self, 원본점들, 회전된점들, 중심점, 각도_도):
        """회전 변환 시각화"""
        print(f"🎨 '아름다운 회전 춤을 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            plt.figure(figsize=(12, 10))
            
            # 원본 점들
            원본_x = [p[0] for p in 원본점들]
            원본_y = [p[1] for p in 원본점들]
            plt.scatter(원본_x, 원본_y, c='blue', s=100, alpha=0.7, label='원본 점들')
            
            # 회전된 점들
            회전_x = [p[0] for p in 회전된점들]
            회전_y = [p[1] for p in 회전된점들]
            plt.scatter(회전_x, 회전_y, c='red', s=100, alpha=0.7, label='회전된 점들')
            
            # 중심점
            plt.scatter(*중심점, c='green', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label='회전 중심', zorder=5)
            
            # 점들 연결 (원본과 회전된 점)
            for i, (원본, 회전) in enumerate(zip(원본점들, 회전된점들)):
                # 원본 점 번호
                plt.annotate(f'P{i+1}', 원본, xytext=(5, 5), 
                           textcoords='offset points', color='blue', fontweight='bold')
                
                # 회전된 점 번호
                plt.annotate(f"P{i+1}'", 회전, xytext=(5, 5), 
                           textcoords='offset points', color='red', fontweight='bold')
                
                # 중심점에서 원본점으로의 선
                plt.plot([중심점[0], 원본[0]], [중심점[1], 원본[1]], 
                        'b--', alpha=0.5, linewidth=1)
                
                # 중심점에서 회전된점으로의 선
                plt.plot([중심점[0], 회전[0]], [중심점[1], 회전[1]], 
                        'r--', alpha=0.5, linewidth=1)
                
                # 회전 호 그리기 (중심점 기준)
                반지름 = ((원본[0] - 중심점[0])**2 + (원본[1] - 중심점[1])**2)**0.5
                
                if 반지름 > 0:
                    시작각 = math.degrees(math.atan2(원본[1] - 중심점[1], 원본[0] - 중심점[0]))
                    끝각 = 시작각 + 각도_도
                    
                    # 호 그리기
                    if abs(각도_도) > 5:  # 5도 이상일 때만 호 표시
                        각도들 = np.linspace(math.radians(시작각), math.radians(끝각), 50)
                        호_x = 중심점[0] + 반지름 * 0.3 * np.cos(각도들)  # 0.3배 크기로 축소
                        호_y = 중심점[1] + 반지름 * 0.3 * np.sin(각도들)
                        plt.plot(호_x, 호_y, 'orange', linewidth=2, alpha=0.7)
            
            # 격자와 축
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linewidth=0.5)
            plt.axvline(x=0, color='k', linewidth=0.5)
            
            plt.title(f'🔄 로테이터의 회전 춤 ({각도_도}도 회전)', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            plt.legend()
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib와 numpy가 필요해!'")

# 실전 예시
print("🔄 로테이터의 회전 춤 모험:")
로테이터 = 로테이터의_회전춤()

# 테스트 점들 (삼각형 모양)
원본점들 = [(3, 1), (5, 1), (4, 3)]
중심점 = (4, 2)  # 삼각형 중심 근처
회전각도 = 45    # 45도 회전

print(f"\n🎭 회전 춤 설정:")
print(f"   원본 점들: {원본점들}")
print(f"   회전 중심: {중심점}")
print(f"   회전 각도: {회전각도}도")

# 회전 방향 설명
로테이터.회전_방향_설명(회전각도)

print(f"\n" + "="*60)
print("💃 회전 춤 시작!")

# 개별 점 회전 예시
print(f"\n🔍 개별 점 회전 예시:")
첫번째점 = 원본점들[0]
회전된_첫번째점 = 로테이터.중심점_회전(첫번째점, 중심점, 회전각도)

# 전체 점들 회전
print(f"\n" + "="*60)
print("🎭 전체 점들 회전:")
회전된점들 = 로테이터.다중점_회전(원본점들, 중심점, 회전각도)

# 특수 각도들로 추가 실험
특수각도들 = [90, 180, 270, -90]

print(f"\n🌟 특수 각도 실험:")
for 각도 in 특수각도들:
    print(f"\n   🔄 {각도}도 회전:")
    특수_회전점 = 로테이터.중심점_회전(첫번째점, 중심점, 각도)

# 시각화
로테이터.시각화(원본점들, 회전된점들, 중심점, 회전각도)
```

**⏰ 시간복잡도**: O(1) - "각 점마다 상수 시간!"
**💫 마법 속성**: 아름다운 회전 변환과 기하학적 변화
**🎯 장점**: 정확한 변환, 임의 중심점 가능
**⚠️ 단점**: 부동소수점 오차, 복잡한 3D 확장

---

## 🌟 제5막: 고급 기하 협회의 대모험

### 🌀 스위퍼 (Sweep Line) - "회전 청소의 달인"

#### 캐릭터 설정 🧹
- **클래스**: 청소부 (Sweeper)
- **정체**: 회전하는 선으로 평면을 깔끔하게 정리하는 기하학적 청소 전문가
- **별명**: "회전선의 마에스트로"
- **성격**: 체계적이고 정확함, 순서를 중시하고 깔끔한 것을 좋아함
- **특기**: 회전 스위핑으로 복잡한 기하 문제를 체계적으로 해결
- **철학**: "체계적으로 훑으면 복잡한 것도 간단해진다"
- **말버릿**: "차근차근 훑어보자!"
- **무기**: 회전 스위프 빗자루 (각도 기반 정렬)

#### 회전 스위프 모험 🌪️
```python
import math

class 스위퍼의_회전청소:
    def __init__(self):
        print("🌀 스위퍼: '안녕! 회전 청소로 기하 문제를 깔끔하게 정리하는 스위퍼야!'")
    
    def 각도_계산(self, 중심점, 목표점):
        """중심점에서 목표점으로의 각도 계산"""
        cx, cy = 중심점
        tx, ty = 목표점
        
        dx, dy = tx - cx, ty - cy
        각도_라디안 = math.atan2(dy, dx)
        각도_도 = math.degrees(각도_라디안)
        
        # 0 ~ 360도 범위로 정규화
        if 각도_도 < 0:
            각도_도 += 360
            
        return 각도_도, 각도_라디안
    
    def 가장_가까운_점쌍_스위프(self, 점들):
        """회전 스위프로 가장 가까운 점 쌍 찾기"""
        print(f"🌀 스위퍼: '회전 스위프로 {len(점들)}개 점에서 가장 가까운 쌍을 찾아보자!'")
        
        if len(점들) < 2:
            return None, float('inf')
        
        최소거리 = float('inf')
        최근접쌍 = None
        
        # 각 점을 중심으로 회전 스위프
        for i, 중심점 in enumerate(점들):
            print(f"\n🎯 중심점 {i+1}: {중심점}")
            
            # 중심점을 제외한 다른 점들
            다른점들 = [p for j, p in enumerate(점들) if j != i]
            
            if not 다른점들:
                continue
            
            # 각 점에 대해 각도와 거리 계산
            점_정보들 = []
            for j, 점 in enumerate(다른점들):
                각도, _ = self.각도_계산(중심점, 점)
                거리 = math.sqrt((점[0] - 중심점[0])**2 + (점[1] - 중심점[1])**2)
                점_정보들.append((각도, 거리, 점, j))
            
            # 각도 순으로 정렬
            점_정보들.sort(key=lambda x: x[0])
            
            print(f"   📐 각도 순 정렬:")
            for 각도, 거리, 점, _ in 점_정보들:
                print(f"      각도 {각도:.1f}°, 거리 {거리:.3f}: {점}")
            
            # 인접한 점들 간의 거리 확인
            for k in range(len(점_정보들)):
                현재_거리 = 점_정보들[k][1]
                현재_점 = 점_정보들[k][2]
                
                if 현재_거리 < 최소거리:
                    최소거리 = 현재_거리
                    최근접쌍 = (중심점, 현재_점)
                    print(f"      🌟 새로운 최근접 쌍! 거리: {최소거리:.3f}")
        
        print(f"\n🎉 회전 스위프 완료!")
        print(f"   👫 최근접 쌍: {최근접쌍}")
        print(f"   📏 최소 거리: {최소거리:.3f}")
        
        return 최근접쌍, 최소거리
    
    def 가시성_스위프(self, 중심점, 장애물들):
        """중심점에서의 가시성 영역을 회전 스위프로 계산"""
        print(f"🌀 스위퍼: '중심점 {중심점}에서의 가시성을 스위프로 분석해보자!'")
        
        if not 장애물들:
            print("   🌟 장애물이 없어서 모든 방향이 보여!")
            return []
        
        # 각 장애물(선분)에 대해 이벤트 생성
        이벤트들 = []
        
        for i, (시작점, 끝점) in enumerate(장애물들):
            시작각도, _ = self.각도_계산(중심점, 시작점)
            끝각도, _ = self.각도_계산(중심점, 끝점)
            
            # 시작각도가 끝각도보다 클 수 있으므로 정렬
            if 시작각도 > 끝각도:
                시작각도, 끝각도 = 끝각도, 시작각도
            
            이벤트들.append((시작각도, 'start', i, 시작점))
            이벤트들.append((끝각도, 'end', i, 끝점))
            
            print(f"   🚧 장애물 {i+1}: {시작점} → {끝점}")
            print(f"      각도 범위: {시작각도:.1f}° ~ {끝각도:.1f}°")
        
        # 각도 순으로 이벤트 정렬
        이벤트들.sort(key=lambda x: (x[0], x[1] == 'end'))
        
        print(f"\n🔄 스위프 라인 진행:")
        활성_장애물들 = set()
        가시_구간들 = []
        마지막_각도 = 0
        
        for 각도, 타입, 장애물_id, 점 in 이벤트들:
            print(f"   📍 각도 {각도:.1f}°: {타입} 장애물 {장애물_id+1}")
            
            # 현재까지가 가시 구간이었다면 기록
            if not 활성_장애물들 and 각도 > 마지막_각도:
                가시_구간들.append((마지막_각도, 각도))
                print(f"      👁️ 가시 구간: {마지막_각도:.1f}° ~ {각도:.1f}°")
            
            if 타입 == 'start':
                활성_장애물들.add(장애물_id)
            else:
                활성_장애물들.discard(장애물_id)
            
            마지막_각도 = 각도
            print(f"      📋 활성 장애물들: {list(활성_장애물들)}")
        
        # 마지막부터 360도까지
        if not 활성_장애물들 and 마지막_각도 < 360:
            가시_구간들.append((마지막_각도, 360))
            print(f"      👁️ 마지막 가시 구간: {마지막_각도:.1f}° ~ 360°")
        
        print(f"\n👁️ 가시성 분석 결과:")
        총_가시_각도 = sum(끝 - 시작 for 시작, 끝 in 가시_구간들)
        가시성_비율 = (총_가시_각도 / 360) * 100
        
        print(f"   📊 가시 구간들: {가시_구간들}")
        print(f"   📊 총 가시 각도: {총_가시_각도:.1f}°")
        print(f"   📊 가시성 비율: {가시성_비율:.1f}%")
        
        return 가시_구간들
    
    def 시각화(self, 중심점, 점들=None, 장애물들=None, 가시_구간들=None, 최근접쌍=None):
        """스위프 결과 시각화"""
        print(f"🎨 '회전 청소 결과를 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import matplotlib.patches as patches
            
            plt.figure(figsize=(12, 10))
            
            # 중심점 표시
            plt.scatter(*중심점, c='red', s=200, marker='*', 
                      edgecolors='black', linewidth=2, label='중심점', zorder=5)
            
            # 다른 점들 표시 (최근접 점 찾기용)
            if 점들:
                for i, 점 in enumerate(점들):
                    if 점 != 중심점:
                        plt.scatter(*점, c='blue', s=100, alpha=0.7)
                        plt.annotate(f'P{i+1}', 점, xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10)
                        
                        # 중심점에서 각 점으로의 선
                        plt.plot([중심점[0], 점[0]], [중심점[1], 점[1]], 
                                'gray', linestyle='--', alpha=0.5)
            
            # 최근접 쌍 강조
            if 최근접쌍:
                점1, 점2 = 최근접쌍
                plt.plot([점1[0], 점2[0]], [점1[1], 점2[1]], 
                        'red', linewidth=3, alpha=0.8, label='최근접 쌍')
            
            # 장애물들 표시
            if 장애물들:
                for i, (시작점, 끝점) in enumerate(장애물들):
                    plt.plot([시작점[0], 끝점[0]], [시작점[1], 끝점[1]], 
                            'black', linewidth=3, alpha=0.8)
                    
                    # 장애물 중점에 번호 표시
                    중점_x = (시작점[0] + 끝점[0]) / 2
                    중점_y = (시작점[1] + 끝점[1]) / 2
                    plt.annotate(f'장애물{i+1}', (중점_x, 중점_y), 
                               ha='center', va='center', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow"))
            
            # 가시성 구간 표시
            if 가시_구간들:
                반지름 = 5  # 가시성 호의 반지름
                
                for 시작각도, 끝각도 in 가시_구간들:
                    # 가시성 호 그리기
                    각도들 = np.linspace(math.radians(시작각도), math.radians(끝각도), 50)
                    호_x = 중심점[0] + 반지름 * np.cos(각도들)
                    호_y = 중심점[1] + 반지름 * np.sin(각도들)
                    
                    plt.fill_between(호_x, 호_y, 중심점[1], alpha=0.3, color='green', label='가시 영역' if 시작각도 == 가시_구간들[0][0] else "")
                    plt.plot(호_x, 호_y, 'green', linewidth=2)
                    
                    # 가시 구간 경계선
                    시작_x = 중심점[0] + 반지름 * math.cos(math.radians(시작각도))
                    시작_y = 중심점[1] + 반지름 * math.sin(math.radians(시작각도))
                    끝_x = 중심점[0] + 반지름 * math.cos(math.radians(끝각도))
                    끝_y = 중심점[1] + 반지름 * math.sin(math.radians(끝각도))
                    
                    plt.plot([중심점[0], 시작_x], [중심점[1], 시작_y], 'green', linewidth=2)
                    plt.plot([중심점[0], 끝_x], [중심점[1], 끝_y], 'green', linewidth=2)
            
            plt.title('🌀 스위퍼의 회전 스위프 분석', fontsize=16)
            plt.xlabel('X 좌표')
            plt.ylabel('Y 좌표')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib와 numpy가 필요해!'")

# 실전 예시
print("🌀 스위퍼의 회전 청소 모험:")
스위퍼 = 스위퍼의_회전청소()

print(f"\n" + "="*60)
print("📍 실험 1: 가장 가까운 점 쌍 찾기")

점들 = [(2, 3), (5, 1), (7, 4), (3, 6), (8, 2)]
print(f"🌟 테스트 점들: {점들}")

최근접쌍, 최소거리 = 스위퍼.가장_가까운_점쌍_스위프(점들)

print(f"\n" + "="*60)
print("👁️ 실험 2: 가시성 분석")

중심점 = (5, 5)
장애물들 = [
    ((3, 3), (7, 3)),  # 수평 장애물
    ((6, 2), (6, 6)),  # 수직 장애물  
    ((2, 6), (4, 8))   # 대각선 장애물
]

print(f"👁️ 관찰 중심점: {중심점}")
print(f"🚧 장애물들:")
for i, (시작, 끝) in enumerate(장애물들):
    print(f"   장애물 {i+1}: {시작} → {끝}")

가시_구간들 = 스위퍼.가시성_스위프(중심점, 장애물들)

# 시각화
스위퍼.시각화(중심점, 점들, 장애물들, 가시_구간들, 최근접쌍)
```

**⏰ 시간복잡도**: O(n log n) - "각도 정렬이 필요해!"
**💫 마법 속성**: 체계적인 회전 스위핑으로 복잡한 기하 문제 해결
**🎯 장점**: 체계적 접근, 다양한 기하 문제에 응용 가능
**⚠️ 단점**: 각도 계산 복잡, 정밀도 문제

---

### 🔬 KD트리나 (K-D Tree) - "차원 분할의 도서관장"

#### 캐릭터 설정 📚
- **클래스**: 도서관장 (Librarian)
- **정체**: k차원 공간을 체계적으로 분할하여 효율적인 검색을 제공하는 지식의 수호자
- **별명**: "차원의 사서"
- **성격**: 매우 체계적이고 논리적, 모든 것을 차원별로 정리하고 분류함
- **특기**: 다차원 공간에서의 빠른 최근접 이웃 검색
- **철학**: "차원을 나누어 정복하면 무한한 공간도 관리할 수 있다"
- **말버릿**: "차원별로 정리해보자!"
- **무기**: 차원 분할 나침반 (k차원 이진 트리)

#### K-D 트리 검색 모험 🌳
```python
import math

class KD노드:
    def __init__(self, 점, 차원, 왼쪽=None, 오른쪽=None):
        self.점 = 점
        self.차원 = 차원  # 분할 기준 차원
        self.왼쪽 = 왼쪽
        self.오른쪽 = 오른쪽

class KD트리나의_차원도서관:
    def __init__(self, 차원수=2):
        self.차원수 = 차원수
        self.뿌리 = None
        print(f"🔬 KD트리나: '안녕! {차원수}차원 공간의 도서관장 KD트리나야!'")
    
    def 거리계산(self, 점1, 점2):
        """두 점 사이의 유클리드 거리"""
        return math.sqrt(sum((a - b)**2 for a, b in zip(점1, 점2)))
    
    def 트리_구축(self, 점들, 깊이=0):
        """K-D 트리 구축"""
        if not 점들:
            return None
        
        현재차원 = 깊이 % self.차원수
        
        print(f"   📚 깊이 {깊이}: {현재차원}차원으로 {len(점들)}개 점을 분할")
        print(f"      점들: {점들}")
        
        # 현재 차원을 기준으로 정렬
        점들_정렬 = sorted(점들, key=lambda p: p[현재차원])
        print(f"      {현재차원}차원 기준 정렬: {점들_정렬}")
        
        # 중앙값 선택
        중앙_인덱스 = len(점들_정렬) // 2
        중앙점 = 점들_정렬[중앙_인덱스]
        
        print(f"      중앙점 선택: {중앙점} (인덱스 {중앙_인덱스})")
        
        # 재귀적으로 왼쪽과 오른쪽 서브트리 구성
        왼쪽_점들 = 점들_정렬[:중앙_인덱스]
        오른쪽_점들 = 점들_정렬[중앙_인덱스 + 1:]
        
        if 왼쪽_점들:
            print(f"      ⬅️ 왼쪽 서브트리: {왼쪽_점들}")
        if 오른쪽_점들:
            print(f"      ➡️ 오른쪽 서브트리: {오른쪽_점들}")
        
        노드 = KD노드(
            점=중앙점,
            차원=현재차원,
            왼쪽=self.트리_구축(왼쪽_점들, 깊이 + 1),
            오른쪽=self.트리_구축(오른쪽_점들, 깊이 + 1)
        )
        
        return 노드
    
    def 도서관_구축(self, 점들):
        """K-D 트리 도서관 구축"""
        print(f"🔬 KD트리나: '{len(점들)}개 점으로 {self.차원수}차원 도서관을 구축하겠다!'")
        self.뿌리 = self.트리_구축(점들)
        print(f"📚 도서관 구축 완료!")
    
    def 최근접_이웃_검색(self, 쿼리점):
        """최근접 이웃 검색"""
        print(f"🔍 KD트리나: '쿼리점 {쿼리점}의 최근접 이웃을 찾아보겠다!'")
        
        if not self.뿌리:
            print("😅 '도서관이 비어있어!'")
            return None, float('inf')
        
        최근접점 = [None]
        최소거리 = [float('inf')]
        
        def 검색_도우미(노드, 깊이):
            if 노드 is None:
                return
            
            현재차원 = 깊이 % self.차원수
            print(f"\n   📖 깊이 {깊이}: 노드 {노드.점} 방문 ({현재차원}차원 분할)")
            
            # 현재 노드와의 거리 계산
            현재거리 = self.거리계산(쿼리점, 노드.점)
            print(f"      📏 거리: {현재거리:.3f}")
            
            # 최근접점 업데이트
            if 현재거리 < 최소거리[0]:
                최소거리[0] = 현재거리
                최근접점[0] = 노드.점
                print(f"      🌟 새로운 최근접점! {노드.점} (거리: {현재거리:.3f})")
            
            # 어느 쪽으로 먼저 탐색할지 결정
            if 쿼리점[현재차원] < 노드.점[현재차원]:
                가까운쪽 = 노드.왼쪽
                먼쪽 = 노드.오른쪽
                방향 = "왼쪽"
            else:
                가까운쪽 = 노드.오른쪽
                먼쪽 = 노드.왼쪽
                방향 = "오른쪽"
            
            print(f"      🎯 {방향}부터 탐색")
            
            # 가까운 쪽 먼저 탐색
            검색_도우미(가까운쪽, 깊이 + 1)
            
            # 다른 쪽도 탐색할 필요가 있는지 확인
            차원간_거리 = abs(쿼리점[현재차원] - 노드.점[현재차원])
            print(f"      🔍 다른 쪽 탐색 필요성 확인: 차원간 거리 {차원간_거리:.3f} vs 최소거리 {최소거리[0]:.3f}")
            
            if 차원간_거리 < 최소거리[0]:
                print(f"      ↔️ 다른 쪽도 탐색 필요!")
                검색_도우미(먼쪽, 깊이 + 1)
            else:
                print(f"      ✂️ 다른 쪽 탐색 불필요 (가지치기)")
        
        검색_도우미(self.뿌리, 0)
        
        print(f"\n🎉 최근접 이웃 검색 완료!")
        print(f"   🎯 쿼리점: {쿼리점}")
        print(f"   👫 최근접점: {최근접점[0]}")
        print(f"   📏 최소거리: {최소거리[0]:.3f}")
        
        return 최근접점[0], 최소거리[0]
    
    def 범위_검색(self, 쿼리점, 반지름):
        """지정된 반지름 내의 모든 점 찾기"""
        print(f"🔍 KD트리나: '쿼리점 {쿼리점} 반지름 {반지름} 내의 모든 점을 찾아보겠다!'")
        
        결과점들 = []
        
        def 범위검색_도우미(노드, 깊이):
            if 노드 is None:
                return
            
            현재차원 = 깊이 % self.차원수
            현재거리 = self.거리계산(쿼리점, 노드.점)
            
            print(f"   📖 깊이 {깊이}: 노드 {노드.점} (거리: {현재거리:.3f})")
            
            # 현재 노드가 범위 내에 있는지 확인
            if 현재거리 <= 반지름:
                결과점들.append(노드.점)
                print(f"      ✅ 범위 내 점 발견!")
            
            # 자식 노드들 탐색 여부 결정
            차원간_거리 = abs(쿼리점[현재차원] - 노드.점[현재차원])
            
            if 차원간_거리 <= 반지름:
                print(f"      ↔️ 양쪽 자식 모두 탐색")
                범위검색_도우미(노드.왼쪽, 깊이 + 1)
                범위검색_도우미(노드.오른쪽, 깊이 + 1)
            elif 쿼리점[현재차원] < 노드.점[현재차원]:
                print(f"      ⬅️ 왼쪽 자식만 탐색")
                범위검색_도우미(노드.왼쪽, 깊이 + 1)
            else:
                print(f"      ➡️ 오른쪽 자식만 탐색")
                범위검색_도우미(노드.오른쪽, 깊이 + 1)
        
        범위검색_도우미(self.뿌리, 0)
        
        print(f"\n🌐 범위 검색 완료!")
        print(f"   📍 찾은 점들: {결과점들}")
        print(f"   📊 점 개수: {len(결과점들)}")
        
        return 결과점들
    
    def 트리_구조_출력(self, 노드=None, 깊이=0, 접두사=""):
        """K-D 트리 구조 시각적 출력"""
        if 노드 is None:
            if 깊이 == 0:
                노드 = self.뿌리
            else:
                return
        
        현재차원 = 깊이 % self.차원수
        print(f"{접두사}📚 깊이{깊이} (차원{현재차원}): {노드.점}")
        
        if 노드.왼쪽 or 노드.오른쪽:
            if 노드.왼쪽:
                print(f"{접두사}├── 왼쪽:")
                self.트리_구조_출력(노드.왼쪽, 깊이 + 1, 접두사 + "│   ")
            
            if 노드.오른쪽:
                print(f"{접두사}└── 오른쪽:")
                self.트리_구조_출력(노드.오른쪽, 깊이 + 1, 접두사 + "    ")
    
    def 시각화(self, 점들, 쿼리점=None, 최근접점=None, 범위점들=None, 반지름=None):
        """K-D 트리와 검색 결과 시각화"""
        print(f"🎨 '차원 도서관을 그려볼게!'")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            plt.figure(figsize=(12, 10))
            
            # 모든 점들 표시
            x_coords = [p[0] for p in 점들]
            y_coords = [p[1] for p in 점들]
            plt.scatter(x_coords, y_coords, c='lightblue', s=100, alpha=0.7, label='모든 점들')
            
            # 점 번호 표시
            for i, (x, y) in enumerate(점들):
                plt.annotate(f'P{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
            
            # 쿼리점 표시
            if 쿼리점:
                plt.scatter(*쿼리점, c='red', s=200, marker='*', 
                          edgecolors='black', linewidth=2, label='쿼리점', zorder=5)
                plt.annotate('Query', 쿼리점, xytext=(10, 10), textcoords='offset points',
                           fontsize=12, fontweight='bold')
            
            # 최근접점 표시
            if 최근접점:
                plt.scatter(*최근접점, c='green', s=150, marker='o', 
                          edgecolors='black', linewidth=2, label='최근접점', zorder=4)
                
                # 쿼리점과 최근접점 연결
                if 쿼리점:
                    plt.plot([쿼리점[0], 최근접점[0]], [쿼리점[1], 최근접점[1]], 
                            'green', linewidth=2, linestyle='--', alpha=0.7)
            
            # 범위 검색 결과 표시
            if 범위점들:
                범위_x = [p[0] for p in 범위점들]
                범위_y = [p[1] for p in 범위점들]
                plt.scatter(범위_x, 범위_y, c='orange', s=120, alpha=0.8, 
                          label='범위 내 점들', zorder=3)
            
            # 검색 반지름 표시
            if 쿼리점 and 반지름:
                원 = patches.Circle(쿼리점, 반지름, fill=False, 
                                  color='red', linestyle='--', linewidth=2, alpha=0.7)
                plt.gca().add_patch(원)
            
            plt.title('🔬 KD트리나의 차원 도서관', fontsize=16)
            plt.xlabel('X 좌표 (0차원)')
            plt.ylabel('Y 좌표 (1차원)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("🎨 '시각화를 위해서는 matplotlib가 필요해!'")

# 실전 예시
print("🔬 KD트리나의 차원 도서관 모험:")
KD트리나 = KD트리나의_차원도서관(2)  # 2차원

# 테스트 점들
점들 = [(3, 6), (17, 15), (13, 15), (6, 12), (9, 1), (2, 7), (10, 4)]

print(f"\n📚 도서관에 보관할 점들: {점들}")

# K-D 트리 구축
KD트리나.도서관_구축(점들)

# 트리 구조 출력
print(f"\n🌳 구축된 K-D 트리 구조:")
KD트리나.트리_구조_출력()

print(f"\n" + "="*60)
print("🔍 실험 1: 최근접 이웃 검색")

쿼리점 = (9, 2)
최근접점, 최소거리 = KD트리나.최근접_이웃_검색(쿼리점)

print(f"\n" + "="*60)
print("🌐 실험 2: 범위 검색")

반지름 = 3
범위점들 = KD트리나.범위_검색(쿼리점, 반지름)

# 시각화
KD트리나.시각화(점들, 쿼리점, 최근접점, 범위점들, 반지름)
```

**⏰ 시간복잡도**: 평균 O(log n), 최악 O(n)
**💫 마법 속성**: 다차원 공간의 효율적인 분할과 검색
**🎯 장점**: 고차원에서도 효율적, 다양한 검색 질의 지원
**⚠️ 단점**: 차원이 높아질수록 성능 저하, 균형 유지 필요

---

## 🌟 제6막: 요정 왕국 기하 올림픽

### 🏆 모든 요정들의 대회전

```python
import time
import random
import math

def 기하_요정_올림픽():
    print("🏆 기하 요정 왕국 올림픽 개막!")
    print("✨ 모든 요정들이 한자리에 모여 실력을 겨룬다!")
    print("=" * 60)
    
    # 테스트 데이터 준비
    random.seed(42)
    점들 = [(random.randint(0, 20), random.randint(0, 20)) for _ in range(10)]
    
    print(f"🌟 올림픽 경기장의 점들: {점들}")
    print()
    
    결과들 = {}
    
    # 🔍 포인티의 가장 가까운 점 찾기
    print("🥇 종목 1: 가장 가까운 친구 찾기 (포인티)")
    시작시간 = time.time()
    
    최소거리 = float('inf')
    for i in range(len(점들)):
        for j in range(i + 1, len(점들)):
            거리 = math.sqrt((점들[i][0] - 점들[j][0])**2 + (점들[i][1] - 점들[j][1])**2)
            if 거리 < 최소거리:
                최소거리 = 거리
    
    포인티_시간 = time.time() - 시작시간
    결과들["🔍 포인티"] = (포인티_시간 * 1000, f"최소거리: {최소거리:.3f}")
    
    # 🌊 컨벡시의 볼록 껍질
    print("🥈 종목 2: 보호막 만들기 (컨벡시)")
    시작시간 = time.time()
    
    # 단순한 볼록 껍질 (기프트 래핑)
    def 외적(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])
    
    시작점 = min(점들, key=lambda p: p[0])
    볼록껍질 = []
    현재점 = 시작점
    
    while True:
        볼록껍질.append(현재점)
        다음점 = 점들[0]
        for 후보점 in 점들[1:]:
            if 외적(현재점, 다음점, 후보점) > 0:
                다음점 = 후보점
        if 다음점 == 시작점:
            break
        현재점 = 다음점
    
    컨벡시_시간 = time.time() - 시작시간
    결과들["🌊 컨벡시"] = (컨벡시_시간 * 1000, f"볼록껍질 점 수: {len(볼록껍질)}")
    
    # 🎯 폴리고니의 내부 판정
    print("🥉 종목 3: 경계 판정 (폴리고니)")
    시작시간 = time.time()
    
    테스트점 = (10, 10)
    사각형 = [(5, 5), (15, 5), (15, 15), (5, 15)]
    
    # 레이 캐스팅
    x, y = 테스트점
    교차횟수 = 0
    
    for i in range(len(사각형)):
        x1, y1 = 사각형[i]
        x2, y2 = 사각형[(i + 1) % len(사각형)]
        
        if min(y1, y2) < y <= max(y1, y2):
            if y1 != y2:
                교차점_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if 교차점_x > x:
                    교차횟수 += 1
    
    내부여부 = 교차횟수 % 2 == 1
    폴리고니_시간 = time.time() - 시작시간
    결과들["🎯 폴리고니"] = (폴리고니_시간 * 1000, f"점 {테스트점}: {'내부' if 내부여부 else '외부'}")
    
    print(f"\n🏁 기하 올림픽 결과:")
    print("-" * 50)
    
    # 결과 정렬 (시간 순)
    정렬된결과 = sorted(결과들.items(), key=lambda x: x[1][0])
    
    메달 = ["🥇", "🥈", "🥉"]
    for i, (이름, (시간, 설명)) in enumerate(정렬된결과):
        메달아이콘 = 메달[i] if i < 3 else "🏅"
        print(f"{메달아이콘} {이름}: {시간:.4f}ms - {설명}")
    
    print(f"\n🎊 모든 기하 요정들이 각자의 특기를 보여주었다!")
    print(f"🌟 가장 중요한 것은 상황에 맞는 올바른 알고리즘을 선택하는 것!")

# 올림픽 개최!
기하_요정_올림픽()
```

---

## 🌟 에필로그: 지오메트리아 요정 왕국의 새로운 전설

*모든 기하 요정들의 모험이 끝나고, 지오메트리아 왕국에 평화가 찾아왔다...*

```
🌅 지오메트리아의 새로운 아침

각 요정들은 자신만의 특별한 마법을 완성했다:

🔍 포인티: "가장 가까운 친구를 찾는 것이 내 기쁨!"
🌊 컨벡시: "모든 것을 포용하는 보호막을 만들어!"
⚡ 인터섹: "선분들의 운명적 만남을 예언하지!"
📐 트라이앵: "가장 아름다운 삼각형으로 세상을 나누어!"
🗺️ 보로노이: "공정한 영토 분할이 평화의 시작!"
🎯 폴리고니: "경계가 명확해야 세상이 질서정연해!"
🔄 로테이터: "모든 변화는 아름다운 춤이야!"
📏 스케일러: "크기를 조절하여 완벽한 비율을!"
🌀 스위퍼: "체계적으로 훑으면 복잡한 것도 간단해!"
🔬 KD트리나: "차원을 나누어 무한 공간을 정복!"

그리고 현명한 기하 요정 왕이 말했다:
"점과 선, 면과 공간이 만드는 모든 아름다움을
이해하고 활용하는 것, 그것이 바로
진정한 기하학적 지혜이다!"
```

**🎓 기하 요정 대모험에서 배운 교훈:**

1. **차원의 이해**: 2D에서 3D, 그리고 고차원으로의 확장
2. **효율성의 중요성**: O(n²)에서 O(n log n)으로의 최적화
3. **기하학적 직관**: 수학적 계산과 시각적 이해의 조화
4. **실용성**: 게임, 컴퓨터 그래픽스, 로보틱스 등 실제 응용
5. **정밀성**: 부동소수점 오차와 특수 케이스 처리의 중요성

**🌟 기하 알고리즘의 실제 응용:**
- 🎮 **게임 개발**: 충돌 감지, 경로 찾기, 렌더링
- 🗺️ **지리정보시스템**: 지도 분석, 최적 경로, 영역 분할
- 🤖 **로보틱스**: 경로 계획, 장애물 회피, 동작 제어
- 🏗️ **컴퓨터 그래픽스**: 3D 모델링, 애니메이션, 렌더링
- 📡 **통신**: 기지국 배치, 커버리지 최적화
- 🧬 **생물정보학**: 단백질 구조 분석, 분자 모델링

**🚀 다음 모험지 예고:**
*기하 요정 왕국의 모험이 끝났다. 다음에는 어떤 알고리즘 세계로 떠날까?*
*그래프 제국의 네트워크 모험? 동적계획법 마을의 최적화 여행?*
*아니면 기계학습 우주의 AI 탐험?*

**✨ 특별 감사:**
*모든 점들과 선분들, 다각형들과 공간들이*
*아름다운 기하학적 조화를 이루어*
*요정들의 모험을 가능하게 해주었다!*

---

🧚‍♀️ **기하 요정 대모험, 완결!** 🌟

*"모든 점은 무한한 가능성이고,*
*모든 선은 새로운 연결이며,*
*모든 면은 아름다운 세계이다!"*

🎉 **끝** 🎉# 🧚‍♀️ 기하 요정 대모험: 점과 선의 마법 세계