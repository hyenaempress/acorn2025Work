# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# 장고로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.
# Create your views here.

from django.shortcuts import render
from .models import Jikwon
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

def index(request):
    jikwon_list = Jikwon.objects.all()
    data = pd.DataFrame(jikwon_list.values("jikwonpay", "jikwonibsail"))
    # 입사일을 근무년수(숫자)로 변환
    today = datetime.today().date()
    data['근무년수'] = data['jikwonibsail'].apply(lambda x: (today - x).days // 365)
    train, test = train_test_split(data, test_size=0.1, random_state=1)
    x_train = train[['근무년수']]
    y_train = train['jikwonpay']
    x_test = test[['근무년수']]
    y_test = test['jikwonpay']

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print('예측값 : ', np.round(y_pred, 0))
    print('실제값 : ', y_test.values)

        
    # # 잔차 구하기
    # y_mean = n = np.mean(y_test)  # y의 평균
    # # 오차 제곱합() : sum(y예측값 - y실제값)^2
    # bunja = np.sum(np.square(y_test - y_pred))  # 예측값과 실제값의 차이의 제곱의 합
    # # 편차 제곱합() : sum(y관측값 - y평균값)^2
    # bunmo = np.sum(np.square(y_test - y_mean))
    # r2 = 1 - bunja / bunmo  # 1 - (오차제곱합 / 편차제곱합)
    # print("계산에 의한 결정계수 : ", r2)

    print("api 제공 메소드 결정계수 : ", r2_score(y_test, y_pred))

    return render(request, "index.html", {"jikwon_list": jikwon_list})


# 2. 데이터를 트레이닝과 테스트로 분할(train_test_split을 활용)
# 3. 회귀분석 모델 학습 (LinearRegression 사용)
# 4. 모델 예측실행 (predict(test_data))
# 5. 모델 평가 (r_squared)
# 6. 만들어진 모델을 저장(joblib)
"""
1. 데이터베이스 읽는다. 
2. 선형회귀 모델 만들어서 검증하기
3. 만든 모델 저장하기
3.1 인덱스함수를 비운다. / 주석처리

4. 인덱스 html에서 ajax로 근무연수 입력받는다.

5. 저장된 선형회귀 모델과 근무연수를 이용해 연봉 예측

6. 예측된 연봉을 ajax로 index.html에 보여준다.
"""