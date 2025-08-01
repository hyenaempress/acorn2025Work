#배열에 행 또는 열 추가
import numpy as np

aa = np.eye(3)  # 3x3 단위 행렬 생성
print("=== 배열에 행 또는 열 추가 ===")
print('aa:\n',aa)

bb = np.c_[aa, aa[2]]
print("bb = np.c_[aa, aa[2]] (행 추가):\n", bb)

cc = np.r_[aa, [aa[2]]]
print("cc = np.r_[aa, [aa(2)]] (열 추가):\n", cc)

#reshape
a = np.array([1, 2, 3])
print('np,c_:\n', np.c_[a])  # 1차원 배열을 2차원으로 변환
#이런식으로 구조가 바뀔 수 있다.
a.reshape(3, 1)  # 3행 1열로 변환
print("=== 배열에 행 또는 열 추가 후 모양 확인 ===")
print(a)  # (3, 4) - 3행 4열

#append insert delete
b= np.append(a, [4, 5])  # 배열에 요소 추가
print(b)
c = np.insert(b, 2, [ 6 , 7 ] )  # 인덱스 2에 99 삽입
print(c)
print("=== 배열에 행 또는 열 추가 후 모양 확인 ===")
print(f"cc.shape: {cc.shape}")  # (4, 3) - 4행 3열
#d = delete(c, 1)  # 인덱스 1의 요소 삭제
#print("d = delete(c, 1) (인덱스 1 삭제):", d)
#1차원 배열에서 한 것 

#2차원 배열에서 행 또는 열 추가
print("=== 2차원 배열에 행 또는 열 추가 ===")

aa = np.arange(1 ,10).reshape(3, 3)  # 3x3 배열 생성
print(aa)
print(np.insert(aa, 1, 99)) #삽입후 차원 축소 
#차원축소 안하는 법 
print(np.insert(aa, 1, 99, axis=0))  # 행 추가 (차원 유지)
print(np.insert(aa, 1, 99, axis=1))  # 열 추가 (차원 유지)
print(np.append(aa, [[99, 99, 99]], axis=0))  # 행 추가
print(np.append(aa, [[99], [99], [99]], axis=1))
#print(np.delete(aa, 1, axis=0))  # 인덱스 1의 행 삭제
#print(np.delete(aa, 1, axis=1))  # 인덱스 1의 열 삭제

print(aa)

bb = np.arange(10, 19).reshape(3,3)
print(bb)
cc = np.append(aa,bb) #추가 후 차원 축소 
print(cc)
cc = np.append(aa, bb, axis=0)
print(cc)

print("np.append 연습")
print(np.append(aa, 88)) 
print(np.append(aa,[[88,88,88]], axis=0))   
print(np.append(aa,[[88],[88],[88]], axis=1))

   

print("np.delete 연습")
print(np.delete(aa,1)) #삭제 후 차원 축소 
print(np.delete(aa,1, axis=0)) #삭제 후 차원 축소
print(np.delete(aa,1, axis=1)) #삭제 후 차원 축소

#조건 연산 where (조건, 참, 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
condData = np.array([True, False, True])
result = np.where(condData, x , y )
print(result)       

aa = np.where(x >= 2)
print(aa) #(array())
print(np.where(x >= 2, 'T' , 'F') )
print(np.where(x >= 2, x , x+100) )

#정규분포를 따르는 데이터 값 만들기
bb = np.random.randn(4 , 4) #정규분포 가우시안 분포 정규분포를 따르는 난수 - 중심 극한 정리 
print (bb)

#셈플이 30개 이상일때 종의 모양을 따르고 있다
print(np.where(bb > 0, 7, bb))

#where조건에 대한 이야기다 

print('배열 결합 또는 분할 스플릿과 컨틴케이트')

kbs = np.concatenate([x,y])
print(kbs)

#배열 분항 

x1 , x2 = np.split(kbs, 2)
print(x1)
print(x2) 

#좌우 상하 분할 
#퍼센트와 버티칼이 있음 

a = np.arange(1, 17).reshape(4 ,4)
print(a)
x1, x2 = np.hsplit(a,2)
print(x1)
print(x2)


x1, x2 = np.vsplit(a,2)
print(x1)
print(x2)

#마지막으로 복원 및 비복권 추출 샘플링할떄, 추출 
print('데이터 복권, 비복원 추축')

#구슬 주머니를 예시로 복원 추출 비복원 추출 

datas =np.array([1,2,3,4,5,6,7])
#복원추출 

for _ in range(5):
    print(datas[np.random.randint(0, len(datas) - 1) ] , end = '') #이걸 

#비복원 추출 
print()

#cncnfgkstn choice()
#복원추출 
print(np.random.choice(range(1 , 45),6)) 

#비복원 추출 
print(np.random.choice(range(1 , 45),6, replace=False)) 

#가중치를 부여한 랜덤 추출 
ar = 'aie book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar , 3, p=[0.1, 0.1,0.1,0.1,0.1,0.1,0.4]))
#god가 나올 확률이 가장 높아집니다 이런식으로 하면 복원입니다. 초이스는 이런 것도 가능 

#비복원 추출 전용함수 sample 이것도 많이 쓰입니다. 

#샘플도 많이 쓰임 
print()
import random #이건 랜던이라 넣어야함 다시 datas 들고옴 -- 비복원 추출 semple 샘플 쓰기도 하고 초이스 쓰기도 할거다 어떻게 다를까?
print(random.sample(datas.tolist(), 5))

#샘플은 랜덤 모듈에 있습니다. 

#핵심은 어레이와 백터 연산을 운용할 수 있다는 것입니다. 더이상의 함수들은 그때그때 적용해도 됩니다.