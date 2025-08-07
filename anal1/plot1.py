# matplotli는 플로팅 모듈입니다 
#다양한 그래프를 그릴 수 있습니다. 아주 다양한 그래프 지원 
#함수 메서드를 지원합니다.

import numpy as np
import matplotlib.pyplot as plt
#나중에서 콜랩에서 실습할 때 다를 수 있습니다. 

plt.rc('font', family='Malgun Gothic') #폰트가 깨질 수 있어서 맑은 고딕을 삽입한다 애플은 애플 고딕있습니다. 이걸 쓰면 음수가 안됩니다...
plt.rcParams['axes.unicode_minus'] = False #마이너스 기호 깨짐 방지  

#이 위의 값은 항상 고정적이라고 생각하면 됩니다. 
#이제 차트를 그리겠습니다.

"""
x =['서울', '인천', '수원'] #이건 리스트 튜플 되는데 SET 은 안됩니다 인덱스가 오류나요 
y = [5, 3, 7]

#02. 틱, x축 y축 범위 지정  틱을 늘립니다.
plt.xlim([-1, 3])
plt.ylim([0, 10])
# 엑시스 피겨 라는 말을 사용합니다 

#03. 틱하나만 볼까요 
plt.yticks(list(range(0,10,3))) #0부터 10까지 3씩 증가하는 값을 틱으로 지정합니다. 

#plt 를 찍고 나면 많은 조건들을 볼 수 있습니다. 

#1. 가장 기본적인 line 차트를 그리겠습니다.
plt.plot(x, y) #이렇게 하면 차트가 그려집니다.  2차원 평면으로 
#plt.show() 
#그런데 참고로 주피터 노트북에서는 %matplotlib inline 이라고 써야 그려집니다. 
#각각 인덱스가 있어서 차트를 그릴 수 있습니다. 서울은 0 인천은 1 수원은 2
#맥플로립은 파이썬 실행을 잠시 멈추어 두기 때문에 ok 는 창이 꺼져야 합니다.  
print('ok')

#4. 다른 기능 
data = np.arange(1 , 11, 2)
print(data) # 1 3 5 7 9  구간은 4가 됩니다. 
plt.plot(data)
#plt.show() # 이렇게 하면 x 축은 구간으로 들어갑니다 

#5
x = [0,1,2,3,4]
for a, b in zip(x, data):
    plt.text(a, b, str(b))
#plt.show()
    
#6 정리 차원에서 하나 더 보겠습니다.
plt.plot(data)
plt.plot(data, data, 'r') #선의 색은 빨간색 
for a, b in zip(data, data):
    plt.text(a, b, str(b))
plt.show()
#구간이 보입니다
 

#SIN 곡선 그리기 
x = np.arange(10)
y = np.sin(x)
print(x,y)
#plt.plot(x,y) 곡선 1 
#plt.plot(x,y, 'bo') #곡선 2 보라색 점  파란색 동그라미 점 
#plt.plot(x,y, 'r+') #곡선 3 빨간색 플러스 점 
plt.plot(x,y, 'go--', linewidth=2, markersize=12) #선의 두께 2 점의 크기 12 
# - 하면 Solid , -- 하면 점선 , : 하면 점선  
# color 는 색상 지정 
# marker 는 점의 모양 지정 
# markersize 는 점의 크기 지정 
# linewidth 는 선의 두께 지정 
# linestyle 는 선의 모양 지정 
# markerfacecolor 는 점의 색상 지정 
# markeredgecolor 는 점의 테두리 색상 지정 
# markeredgewidth 는 점의 테두리 두께 지정 
# markerfacecolor 는 점의 색상 지정 
# markeredgecolor 는 점의 테두리 색상 지정 
# markeredgewidth 는 점의 테두리 두께 지정 
# markerfacecolor 는 점의 색상 지정 
# markeredgecolor 는 점의 테두리 색상 지정 
#이런 옵션이 엄청 많습니다. 
plt.show()
"""
#홀드 명령 : 하나의 영역에 두개이상의 그래프를 표시 
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x) #sin 곡선 
y_cos = np.cos(x) #cos 곡선 
 
plt.figure(figsize=(10,5)) #그래프 크기 지정 
plt.plot(x, y_sin, 'r') # 직선 
plt.plot(x, y_cos, 'b')  # 산점도 
plt.xlabel('x 축')
plt.ylabel('y 축')
plt.title('제목: sin & cos')
plt.legend(['sin', 'cos']) #범례를 줄 수 있음 
#범례란 무엇인가? 
#범례 지정은 나중에 하겠다 
plt.show()

# 지금은 하나의 공간에다가 두개를 그렸는데 피겨 영역을 나눌 수 있습니다. 서브 플랏이라고 합니다.


# 서브 플랏  subplot  : Fifure 를 여러개 선언한다 
#피겨란 ? 

plt.subplot(2,1,1) # 2행 1열 1번째 영역을 나누는 것입니다. 
plt.plot(x,y_sin)
plt.title('사인')

plt.subplot(2,1,2) # 2행 1열 2번째 영역을 나누는 것입니다. 
plt.plot(x,y_cos)
plt.title('코사인')
plt.show()


#오후 수업 시작 (2시 30분)

print()
irum = ['a', 'b', 'c', 'd', 'e']
kor = [ 80, 50, 70, 70, 90]
eng = [ 60, 80, 70, 60, 90]
plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'bo-')
plt.ylim([0, 100])
plt.legend(['국어', '영어'])
plt.grid(True)
fig = plt.gcf()
plt.show()
fig.savefig('result.png') #이런식으로 데이터 저장도 가능합니다. 

#PNG로 저장하면 동적이니까 
#자바 스크립 라이브러리로 출력도 가능 
#chart.js (장고 등을 이용해서 뽑아 줄 수 있습니다) 이 이외로도 많
#게시글은 토스트 유아이도 있습니다

from matplotlib.image import imread 
#이걸로 읽을수있음
img = imread('result.png')
plt.imshow(img)
plt.show()
#이런식으로 이미지를 읽을 수도 있습니다. 

#그래프 종류가 많잖아요 뭘 꺼내야 할지 잘 모르겠죠





