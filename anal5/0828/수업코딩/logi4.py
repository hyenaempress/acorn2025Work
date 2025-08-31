from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
from matplotlib.colors import ListedColormap
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # 다중 클래스(종속 변수, label, y, class) 지원
plt.rcParams['axes.unicode_minus'] = False


iris = datasets.load_iris()
# print(iris['data'])
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])) # 0.96286

x = iris.data[:, [2, 3]] # petal.length, petal.width만 참여. matrix
y = iris.target # vector

print(x[:3])
print(y[:3], set(y))

# train / test split (7:3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (105, 2) (45, 2) (105,) (45,)

# --------------------------------------------------------------
# # Scaling - Standardization (데이터 표준화 - 최적화 과정에서 안정성, 수렴 속도 향상, 오버플로우, 언더플로우 방지 효과가 적용된 모델을 만든다.)
# print(x_train[:3])
# sc = StandardScaler()
# sc.fit(x_train) # train 데이터로 표준화 객체 생성
# sc.fit(x_test) # test 데이터로 표준화 객체 생성 (이건 안하는게 맞음)
# x_train = sc.transform(x_train)
# x_test = sc.transform(x_test) # 독립 변수만 스케일링
# print(x_train[:3])
# # 스케일링 원복
# inver_x_train = sc.inverse_transform(x_train)
# print(inver_x_train[:3])
# --------------------------------------------------------------

# 분류 모델 생성(데이터 기반 모델 학습)
# 대문자 C 속성: L2 규제 - 모델에 패널티 적용(tuning parameter 중 하나). 숫자 값을 조정해가며 분류 정확도를 확인 1.0, 10.0, 100.0 ... 값이 작을수록 더 강한 정규화 규제를 지원함
model = LogisticRegression(C=0.1, random_state=0, verbose=0).fit(x_train, y_train)
print(model)
model.fit(x_train, y_train) # supervised learning

# 분류 예측 - 모델 성능 파악용
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
print('실제값 : ', y_test)

print('총 갯수:%d, 오류 수: %d' % (len(y_test), (y_test != y_pred).sum())) # 총 갯수:45, 오류 수: 1
print('분류 정확도 확인 1 : ')
print("%.5f" % accuracy_score(y_test, y_pred)) # 1.0

print('분류 정확도 확인 2 : ')
con_mat =  pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값']) # 1.0
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

print('분류 정확도 확인 3 : ')
print('test : ', model.score(x_test, y_test)) # 0.977
print('train :', model.score(x_train, y_train)) # 두 개의 값 차이가 크면 과적합 의심 0.942

# 모델 저장
# pickle.dump(model, open('./01_big_data_machine_learning/data/logistic_model.sav', 'wb'))
del model

read_model = pickle.load(open('./01_big_data_machine_learning/data/logistic_model.sav', 'rb'))

# 읽어온 모델에 대한 정보 확인
print("분류 클래스 개수:", len(read_model.classes_))
print("클래스 목록:", read_model.classes_)

# 새로운 값으로 예측 : petal.length, petal.width 만 참여
# print(x_test[:3])
# [[5.1 2.4]
#  [4.  1. ]
#  [1.4 0.2]]

new_data = np.array([[5.1, 1.1], [1.1, 1.1], [6.1, 7.1]])
# 참고: 만약 표준화한 데이터로 모델을 생성했다면,
# sc.fit(new_data); new_data = sc.transform()
new_pred = read_model.predict(new_data) # 내부적으로 softmax가 출력한 값을 argmax로 처리
print('예측 결과 : ', new_pred) # [2 0 2]
print('예측 결과(확률) : \n', read_model.predict_proba(new_data)) # 각 클래스에 속할 확률
#  [[3.05859126e-02 4.80004878e-01 4.89409209e-01] # softmax가 출력한 값
#  [8.86247468e-01 1.10956949e-01 2.79558303e-03]
#  [1.40841977e-05 4.79092000e-03 9.95194996e-01]]
print("softmax의 총합:", np.sum(read_model.predict_proba(new_data), axis=1)) # softmax의 총합: [1. 1. 1.]

# 시각화
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함 # 왜 5개인가? -> 3개의 클래스[0, 1, 2]가 있으나, 여유있게 5개 정의
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')

    # 1. 색상 팔레트 설정
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # print(cmap.colors[0], cmap.colors[1])
    
    # 데이터 확인
    print("X: \n", X)

    # 2. [면 그리기] surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # (전체 데이터 기준) 첫 번째 feature(꽃잎길이)의 최소/최대값
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # (전체 데이터 기준) 두 번째 feature(꽃잎너비)의 최소/최대값
    
    # 3. [면 그리기] 격자 좌표 생성(뜨문 뜨문 떨어진 점 사이를 촘촘하게 매꿔 면처럼 보이기 위함)
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), np.arange(x2_min, x2_max, resulution))
    
    # 데이터 확인
    print("xx: \n", xx) # 2차원 데이터
    print("yy: \n", yy) # 2차원 데이터
    print("차원 축소 후 전치 결과: \n", np.array([xx.ravel(), yy.ravel()]).T) # 1차원 2줄을 행 2열로 만들고 T를 이용하여 열 2열로 바꿈
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    print("Z: \n", Z)
    print("Z의 차원: ", Z.shape) # (86900,)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원
    print("원래 배열 모양으로 복원한 Z: \n", Z)
    print("Z의 차원: ", Z.shape) # (220, 395)
    

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap) # 면을 그려냄
    plt.xlim(xx.min(), xx.max()) # x축 범위 설정
    plt.ylim(yy.min(), yy.max()) # y축 범위 설정

    for idx, cl in enumerate(np.unique(y)): # 각 클래스에 대해 index와 class를 추출
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), marker=markers[idx], label=cl) # 각 클래스에 대한 산점도
    if test_idx: # test 샘플의 인덱스가 주어지면
        plt.scatter(x=X[test_idx, 0], y=X[test_idx, 1], color=[], marker='o', linewidths=1, s=80, label='test') # test 샘플에 대한 산점도
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
print("x_train: \n", x_train)
print("x_test: \n", x_test)
print("x_combined_std: \n", x_combined_std)
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, test_idx = range(100, 150), title='scikit-learn 제공')