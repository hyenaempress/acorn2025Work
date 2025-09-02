from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 한글 깨짐 방지, 음수 깨짐 방지
plt.rc('font', family = 'Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 사이킷 런에 들어잇는 아이리스 학습용 데이터 로드
iris = datasets.load_iris()
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))  # 상관계수 확인 - 교호작용 유의미성 판단
x = iris.data[:,[2,3]]      # petal.length, petal.width만 참여
y = iris.target
print(x[:3])
print(y[:3], set(y))
print(iris)

# train / test split(7:3)
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)
print('train, test shape 확인 ', x_train.shape, x_test.shape,y_train.shape,y_test.shape)
# (105, 2) (45, 2) (105,) (45,)
"""
# -------------------------------------------------
# scaling 한번 해보자 (데이터 표준화) 데이터 정규화가 아닌 데이터 표준화
# - 최적화 과정에서 안정성, 수렴 속도 향상, 오버플로우/언더플로우 방지 효과
print('-' * 100)
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)       # 독립변수만 스케일링하는거야 종속변수는 하면 안돼!!!!!!!!!!!!
print(x_train[:3])

# 먼저 스케일링은 왜하는거야? 
# 독립변수의 데이터값이 너무 들쭉날쭉일때 
# 표준화를 통해 각 특성을 비슷한 크기로 맞춰줌


# 스케일링 원상복구도 한번 해보자
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
# -------------------------------------------------
"""

# 분류 모델 생성
# model = LogisticRegression(solver = 'lbfgs', C = 0.1, random_state = 0, verbose= 0)   

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy',\
                               max_depth = 5,
                               min_samples_leaf = 5,
                               random_state = 0)

# 이 lbfgs는 디폴트값이야
# 이 C 속성은 L2 규제 - 모델에 패널티 적용.(tuning parameter 중 하나)
# 숫자값을 조정해가며 분류 정확도를 확인한다
# 1.0, 10.0, 100.0, . . ., 값이 작을수록 더 강한 정규화 규제를 가한다.
print(model)
# 이제 학습을 진행시켜보자
model.fit(x_train,y_train)  # supervised learning 사람이 개입해서 학습시키는거야
# 그니까 학습지에 대한 답지를, y_train 값을 같이 줘서 그 방법대로 학습하게 하는거지

# 분류 예측 - 모델 성능 파악용
y_pred = model.predict(x_test)
# 문제를 학습시킬때는 학습지와 답지(x_train, y_train)을 주지만
# 성능을 파악해볼 때는 문제지만(x_test) 줘보는거야

print('예측값 : ',y_pred)
print('실제값 : ',y_test)
print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))
print('-' * 100)

print('분류정확도 확인1 : accuracy score')
print('%.3f'%accuracy_score(y_test, y_pred))
print('-' * 100)

print('분류정확도 확인2 : confusion maxrix')
con_mat = pd.crosstab(y_test,y_pred, rownames = ['예측값'], colnames = ['관측값'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))
print('-' * 100)

print('분류정확도 확인3 : ')
print('test : ', model.score(x_test,y_test))
print('train : ', model.score(x_train,y_train))
print('-' * 100)
# 두 개의 값 차이가 크면 과적합 의심

# 모델 저장
pickle.dump(model, open('logimodel.sav', 'wb'))
del model

# 모델 저장 - 규택 데이터
# pickle.dump(model, open('./logistic_model.sav', 'wb'))
# del model

read_model = pickle.load(open('logimodel.sav', 'rb'))

# 새로운 값으로 예측 : petal.length, petal.width만 참여
# print(x_test[:3])
# 임의의 꽃잎 길이를 줘보고 이게 뭔지 맞추게 해보자
new_data = np.array([[5.1, 1.1],
                     [1.1, 0.1],
                     [6.1, 2.1]])
# 참고 : 만약 표준화한 데이터로 모델을 생성했다면 
# sc.fit(new_data); new_data = sc.transform(new_data)

new_pred = read_model.predict(new_data) # 내부적으로 softmax가 출력한 값을 argmax로 처리

print('예측 결과 : ', new_pred)
print(read_model.predict_proba(new_data))
# softmax가 출력한 값
# [[0.03058591 0.48000488 0.48940921]
#  [0.91507262 0.08391053 0.00101684]
#  [0.00219283 0.15721707 0.8405901 ]]

# 시각화
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                         np.arange(x2_min, x2_max, resulution))
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, \
                  test_idx = range(100, 150), title='scikit-learn 제공')



# 여기 위로는 전에 햇던 아이리스갖고 햇던거고 아래는 9/1에 추가한 내용


# 트리 형태의 시각화
from sklearn import tree
from io import StringIO
import pydotplus

dot_data = StringIO()   # 파일 흉내를 내는 역할
tree.export_graphviz(read_model, out_file = dot_data,
                     feature_names = iris.feature_names[2:4])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('mytree.png')

from matplotlib.pyplot import imread
img = imread('mytree.png')
plt.imshow(img)
plt.show()
plt.close()
# 이거 graphviz 깔아야돼 깔고 패스 설정까지 해줘야된다 핖 인스톨만으로 안된다더라
# https://www.npackd.org/p/org.graphviz.Graphviz/2.38 여기서 다운받으면 돼
# 이걸 설명하는 카페 게시글은 13번 게시글
