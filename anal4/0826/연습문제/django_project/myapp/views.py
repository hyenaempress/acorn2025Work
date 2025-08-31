from django.shortcuts import render
from .models import Jikwon  # Jikwon 모델 임포트
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import date
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error  # 회귀 평가 지표 함수들
from django.http.response import HttpResponse, JsonResponse
import joblib
import json

# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# 장고로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.

# 1. 요구사항 분석
# 종속변수: 연봉, 독립변수: 근무년수

def index(request):
    if request.method == "POST":
        # AJAX로 근무년수 입력 받음
        data = json.loads(request.body)
        years_of_service = float(data.get('years_of_service', 0))

        # 저장된 모델 불러오기
        model_path = "C:/github_personal/HYUNDAI-ROTEM-PARKYK/01_big_data_machine_learning/code/0826/연습문제/django_project/LinearRegression_model.pkl"
        model = joblib.load(model_path)

        # 예측
        predicted_salary = model.predict([[years_of_service]])[0]
        predicted_salary = int(round(predicted_salary, 0))

        return JsonResponse({'salary': predicted_salary})

    # GET 요청 시 기본 페이지 렌더링
    return render(request, "index.html")
    

def make_linear_model(request):
        # 데이터 확인
    jikwon_list = Jikwon.objects.all()
    # for j in jikwon_list:
    #     print(j.jikwonno, j.jikwonname, j.jikwonjik, j.jikwonpay)

    # 1. 필요한 데이터만 추출(연봉, 근무 년수)
    jikwon_list = pd.DataFrame(jikwon_list.values('jikwonpay', 'jikwonibsail'))
    print(jikwon_list)

    # 2. 입사일을 근무 년수로 변환
    today = date.today()
    jikwon_list['years_of_service'] = jikwon_list['jikwonibsail'].apply(lambda x: today.year - x.year if pd.notnull(x) else None)

    # 연봉, 근무년수만 추출
    jikwon_list = jikwon_list[['jikwonpay', 'years_of_service']]
    print(jikwon_list)

    # 3. 데이터 분할 : train, test - sort하면 안돼(왜곡된 자료로 분리)
    train, test = train_test_split(jikwon_list, test_size=0.4, random_state=1)
    print(len(train), len(test))

    x_train = train[['years_of_service']] # (독립 변수) 2차원 데이터 형태
    y_train = train['jikwonpay'] # (종속 변수) 1차원 데이터 형태
    x_test = test[['years_of_service']]
    y_test = test['jikwonpay']

    print("x_train: \n", x_train)
    print("y_train: \n", y_train)
    print("x_test: \n", x_test)
    print("y_test: \n", y_test)

    # 3. 회귀분석 모델 학습(x_train, y_train)
    model = LinearRegression().fit(x_train, y_train) # (독립, 종속)

    # 데이터 확인용
    print("기울기 : ", model.coef_) # [615.5561674]
    print("절편 : ", model.intercept_) # -1707.8469162995625

    # 6. 모델 예측(입력: x_test, 실제 값: y_test)
    prediction_result = model.predict(x_test) # 모델 평가(예측)는 test data를 사용
    print("예측 결과 : ", np.round(prediction_result[:5])) # [6910. 5679. 3832. 1985. 3832.]
    print("실제 결과 : ", y_test[:5].values.round(0)) # [7800 5850 3900 2900 4000]

    # 7. 모델 평가
    # - **R² (결정계수, Coefficient of Determination)** | 0.7020251806664877
    print('R^2_score(결정계수) : {}'.format(r2_score(y_test, prediction_result))) # 절대적으로 사용
    # 기준: 1에 가까울 수록 좋음
    # 해석: 모델이 실제 데이터의 분산 중 70% 설명

    # - **MAE (Mean Absolute Error, 평균 절대 오차)** | 591.4922907488989
    print('MAE(평균절대오차) : {}'.format(np.mean(np.abs(y_test - prediction_result))))
    # 기준: 작을 수록 좋음.

    # - **MSE (Mean Squared Error, 평균 제곱 오차)** | 674271.492221007
    print('MSE(평균제곱오차) : {}'.format(mean_squared_error(y_test, prediction_result)))
    # 기준: 작을 수록 좋음.

    # - **RMSE (Root Mean Squared Error, 평균 제곱근 오차)** | 821.1403608525202
    print('RMSE(평균제곱근오차) : {}'.format(np.sqrt(mean_squared_error(y_test, prediction_result))))
    # 기준: 작을 수록 좋음.

    # 설명분산점수 | 0.7408871511149311
    print('설명분산점수: {}'.format(explained_variance_score(y_test, prediction_result)))
    # 기준: 1에 가까울 수록 좋음

    # 결론: 만들어진 모델은 높은 설명력을 가지고 있음.

    joblib.dump(model, './LinearRegression_model.pkl')  # 모델 저장
    return render(request, "index.html", {"jikwon_list": jikwon_list})