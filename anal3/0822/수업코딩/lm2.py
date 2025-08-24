# https://cafe.daum.net/flowlife/SBU0/29
# 선형회귀 모델식 계산 - 최소제곱법(ols)으로 w=wx+b 형태의 추세식 파라미터 w와 b를 추정한다
# 최소제곱법

import numpy as np

class MySimpleLinearRegression:
    #c Constructor만들기
#     self는 클래스 내부에서 자기 자신(현재 생성된 객체)을 가리키는 예약어예요.
# 👉 즉, self.w = None은 "이 객체가 가진 속성 w를 None으로 초기화한다"는 뜻입니다.
    def __init__(self):
        # 두개의 프로토타입 만듦
        self.w=None
        self.b=None 
        
    #수식으로 만듦
    def fit(self,x:np.ndarray,y:np.ndarray): # x는 독립변수 y는 종속변수 
        # ols로 w,b를 추정
        x_mean=np.mean(x)
        y_mean=np.mean(y)
        numerator=np.sum((x-x_mean)*(y-y_mean)) # 분자
        # 수직적용
        denominator=np.sum((x-x_mean)**2)
        #기울기구하기
        self.w=numerator/denominator
        #절편구하기
        self.b=y_mean-(self.w*x_mean) 
        
#예측값얻기
    def predict(self,x:np.ndarray):
        # 임의의x에 대한 y값은 뭐? 
        return self.w * x+self.b

def main():
    np.random.seed(42)
    # 임의의 성인남성 10명의 키, 몸무게 자료를 사용
    x_heights=np.random.normal(175,5,10)
    y_weights=np.random.normal(70,10,10)

    #최소 제곱법을 수행하는 클래스 객체를 생성 후 학습
    model=MySimpleLinearRegression()
    model.fit(x_heights, y_weights)

    # 추정된 w, b 출력( y = wx + b )
    print('w: ', model.w) # w: -0.23090100700107954

    print('b: ', model.b) # b: 103.0183826888111

    # 예측 값 확인
    y_pred = model.predict(x_heights)
    print('예측 값: ', y_pred)

    print('실제 몸무게와 예측 몸무게의 차이: ', y_weights - y_pred)
    for i in range(len(x_heights)):
        print(f'키: {x_heights[i]:.2f}cm, 실제 몸무게: {y_weights[i]:.2f}kg, 예측 몸무게: {y_pred[i]:.2f}kg')

    print("미지의 남성 키: 199의 몸무게는?", model.predict(199))

    

if __name__=="__main__":
    main()