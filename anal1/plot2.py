import numpy as np
import matplotlib.pyplot as plt



X = np.arange(10)

"""
#여기서 figure 피겨를 형성을 하는데요 방법이 두가지가 있습니다. 
#1) matplotlib 스타일의 인터페이스 

plt.figure()
plt.subplot(2,1,1) #row ,colum , panel number 
plt.plot(x.np.sin(x))
plt.subplot(2,1,2)
plt.plot(x, np.cos(x))
plt.show()

#2) 객체지향 인터페이스 

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(X, np.sin(X))
ax[1].plot(X, np.cos(X))
plt.show()

경웅에 따라서 첫번째 방법을 쓸수도 있고 두번쨰 방법을 쓸 수도 있다.
"""
"""
#첫번째 방법으로 

fig = plt.figure() #명시적으로 선언
ax1 = fig.add_subplot(1,2,1) #1행 2열 1번째 영역을 나누는 것입니다. 
ax2 = fig.add_subplot(1,2,2) #1행 2열 2번째 영역을 나누는 것입니다.  
ax1.hist(np.random.randn(10), bins=5, color='blue', alpha=0.5) #구간 빈즈 투명도 알파 컬러 블루 
ax2.plot(np.random.randn(10))
plt.show() #히스토그램 연속형 데이터를 표현할때 이런 차트를 쓰면 좋습니다. 히스토 그램을 그릴 수 있습니다.

#bar 막대 그래프 그리기
data = [50, 80, 100, 70, 90]
plt.bar(range(len(data)), data)
plt.show()
#세로막대 그래프 그렸습니다.
"""

data = [50, 80, 100, 70, 90]
plt.bar(range(len(data)), data)
plt.show()


#가로 막대 구조 
loss = np.random.rand(len(data))
plt.barh(range(len(data)), data)
plt.show()

#5차막대 그리기 
loss = np.random.rand(len(data))
plt.barh(range(len(data)), data, xerr=loss, alpha=0.7)
plt.show()

#pie

plt.pie(data, explode=[0,0.1,0,0,0], colors=['red', 'blue', 'yellow', 'purple'])
plt.show() 

#지금까지 있는 표는 다량의 데이터를 표기하기는 어려운 지점이 있습니다. 

#제일 중요한건 boxplot 입니다. : 4분위수에 등에 의한 데이터 분포 확인에 효과적입니다. 
plt.boxplot(data)
plt.show()
#최적값 최고값이 나옵니다. 
#나머지는 이상치로 보는 법, 보이는 것은 모든 데이터의 중앙값 

#후딱 지나가요 ~ 
#버블 차트를 봅니다 
#데이터의 크기에 따라 버블의 크기도 변경됩니다. 

n =30
np.random.seed(0)
x = np.random.rand(n)
y = np.random.rand(n)
colors = np.random.rand(n)
scale =np.pi * (15 * np.random.rand(n)) ** 2 
plt.scatter(x, y, s=scale, c=colors, alpha=0.5) #엄청 많이 보게 됩니다.
plt.show()
#데이터의 크기에 따라서 원의 크기가 커지는 버블 차트가 생성됩니다 
#데이터가 무엇이냐에 따라서 조정이 가능합니다. 

#시계열 데이터 만들어 봅니다

import pandas as pd

fdata = pd.DataFrame(np.random.randn(1000, 4),
                     index=pd.date_range('1/1/2000', periods=1000),
                     columns=list('ABCD'))
plt.plot(fdata.cumsum())
plt.show()

#이렇게 사용 할 수도 있습니다. 

fdata = fdata.cumsum() #누적 합계 
print(fdata.head(3))



