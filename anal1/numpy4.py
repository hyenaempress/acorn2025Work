#브로드캐스팅 Broadcasting
#크기가 다른 배열간의 연산 식 배열의 구조 자동 변환이 이루어짐 
#작은 배열과 큰배열 연산 시 작은 배열은 큰배열에 구조를 따름 
import numpy as np

print("=== 브로드캐스팅 예제 ===")
x = np.arange(1, 10).reshape(3, 3)  # 1~9까지 3x3 배열 생성
y = np.array([1, 0, 1])  # 1차원 배열
print(x)
print(y)

#넘파이 브로드캐스팅 안쓸때 
#두 배열의 요소 더하기 
#1) 새로운 배열을 이용 
print("❌ 브로드캐스팅 없이:")
z = np.empty_like(x)  # x와 같은 모양의 빈 배열 생성
print(z)

for i in range(3):
    z[i] = x[i] + y  # 각 행에 y를 더함
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

a= np.array([[0, 1, 2]])
b =np.array([5, 5, 5])
print("브로드캐스팅을 이용한 덧셈:")
print(a + b)  # a와 b를 더함
print(a + 5)  # 브로드캐스팅을 이용한 덧셈

#넘파이로 파일 입출력을 할 수 있다 파일 i/o
print("=== 파일 입출력 ===")
np.save('numpy4etc', x)  # 배열 x를 'data.npy' 파일로 저장
#자동으로 바이너리 형식으로 저장 
np.savetxt('numpy4etc.txt', x)  # 배열 x를 'data.txt' 파일로 저장
imsi = np.load('numpy4etc.npy')  # 'data.npy' 파일에서 배열을 불러옴
print("불러온 배열 imsi:")
print(imsi)  # 불러온 배열 출력
imsi_txt = np.loadtxt('numpy4etc.txt')  # 'data.txt'
mydatas =np.loadtxt('numpy4etc2.txt', delimiter=',')  # 쉼표로 구분된 데이터 불러오기

#구분자가 없는 경우에는 델리미터를 벗어주지 않아도 된다
#일반적으로는 콤마가 많이 붙어있다.

        