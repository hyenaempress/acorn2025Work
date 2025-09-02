# 의사 결정 나무(Decision Tree) - CART (Classification And Regression Tree)
# 예측 분류 모두 가능하나 분류가 주 목적이다
# 비모수검정 : 선형성, 정규성, 등분산성 가정 필요없음
# 유의수준 판단 기준이 없다. 과적합으로 예측 정확도가 낮을 수 있다.

# 키와 머리카락 길이로 성별 구분 모델 작성

import matplotlib.pyplot as plt
from sklearn import tree

# x = [[180, 15],[177, 42],[156, 35],[174, 65],[161, 28]]     # height, hair 
# 1번 해보고 좀 더 복잡한 x데이터로 다뤄보자
x = [[180, 15],[177, 42],[156, 35],[174, 65],[161, 28],[160, 5],[170, 12],[176, 75],[170, 22],[175, 28]]
y = ['man','woman','woman','man','woman','woman','man','man','man','woman']

print(len(x), len(y))
feature_names = ['height','hair length']
class_names = ['male','female']

model = tree.DecisionTreeClassifier(criterion = 'entropy', 
                                    max_depth=5, random_state = 0)  # 'gini'
# 원래 실무에서는 랜덤 스테잇 주면 안된대 근데 지금은 실습이니까

# max_depth: 트리의 최대 깊이를 지정
# min_samples_split:        노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합 제어 가능
# min_samples_leaf:         말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 수. 과적합 제어 가능
# min_weight_fraction_leaf: 말단 노드(leaf)에 있어야 하는 전체 샘플 가중치 합의 최소 비율
# max_features:             최적의 분할을 위해 고려할 최대 feature 수

# DecisionTreeClassifier 클래스에서 선언해줘야되는 아규먼트들인가본데 많이 중요하대
model.fit(x,y)

print('훈련 데이터 정확도 : {:.3f}'.format(model.score(x,y)))
print('예측 결과   : ', model.predict(x))
print('실제 데이터 : ', y)

# 새로운 자료로 분류 예측
new_data = [[199, 60]]
print('예측 결과 : ', model.predict(new_data))

# 시각화
plt.figure(figsize = (10, 6))
tree.plot_tree(model, feature_names = feature_names, \
               class_names=class_names, filled = True, rounded = True, fontsize = 12)
plt.show()
# 이 모델에서는 height를 가장 중요한 요소로 판단한 것 같애 키크니까 다 남자래