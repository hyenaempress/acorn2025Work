#브로드캐스팅 Broadcasting
#크기가 다른 배열간의 연산 식 배열의 구조 자동 변환이 이루어짐 
#작은 배열과 큰배열 연산 시 작은 배열은 큰배열에 구조를 따름 
import numpy as np

print("=== 브로드캐스팅 예제 ===")
x = np.array(1,10).reshape(3, 3)  # 2행 5열 배열
y = np.array([1, 0, 1])  # 1차원 배열
print(x)
print(y)

#넘파이 브로드캐스팅 안쓸때 
#두 배열의 요소 더하기 
#1) 새로운 배열을 이용 
print("❌ 브로드캐스팅 없이:")
z - np.empty_like(x)  # x와 같은 모양의 빈 배열 생성
print(z)

for i in range(3):
    z[i] - x[i] + y  # 각 행에 y를 더함
print("z:")
print(z)

#tile 함수를 사용하는 방법 

kbs = np.tile(y, (3, 1))  # y를 3행으로 반복하여 새로운 배열 생성
print("kbs:")
print(kbs)  
z= x + kbs  # x와 kbs를 더함
print("z:")     
print(z)


print("✅ 브로드캐스팅 사용:")
#타일 메서드를 이용한 것과 비슷하게 브로드캐스팅을 사용하여 연산
kbs = x + y  # x와 y를 더함
print("kbs:")       
print(kbs)  # 브로드캐스팅을 이용한 덧셈 결과
print("브로드캐스팅을 사용하면 코드가 간결해지고, 성능이 향상됩니다.")
print("브로드캐스팅은 크기가 다른 배열 간의 연산을 가능하게 해줍니다.")
print(x + y)  # x와 y를 더함
print("브로드캐스팅 결과:")
print(x + y)  # 브로드캐스팅을 이용한 덧셈
print("브로드캐스팅은 크기가 다른 배열 간의 연산을 가능하게 해줍니다.")
print("브로드캐스팅은 작은 배열을 큰 배열의 구조에 맞춰 자동으로 변환하여 연산을 수행합니다.")
