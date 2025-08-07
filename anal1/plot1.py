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

#가장 기본적인 line 차트를 그리겠습니다.

x =['서울', '인천', '수원'] #이건 리스트 튜플 되는데 SET 은 안됩니다 인덱스가 오류나요 
y = [5, 3, 7]

plt.plot(x, y) #이렇게 하면 차트가 그려집니다.  2차원 평면으로 
plt.show() 
#그런데 참고로 주피터 노트북에서는 %matplotlib inline 이라고 써야 그려집니다. 






