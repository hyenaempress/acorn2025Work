# 앙상블(Ensemble)
# 하나의 샘플데이터를 여러 개의 분류기를 통해 다수의 학습모델을 만들어
# 학습시키고 학습결과를 결합함으로써 과적합을 방지하고 정확도를 높이는 학습기법 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression     #이름은 리그레션이지만 classifier래
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
x, y = cancer.data, cancer.target
print(x[:2],y[:2], set(cancer.target))      # 0 : 음성, 1 : 양성

x_train, x_test, y_train, y_test = train_test_split(x, y, \
                                                    test_size=0.2, random_state=12, stratify = y)

# stratify = y : 레이블 분포가 train/test 고르게 유지하도록 층화(leveling) 샘플링
# 불균형 데이터에서 모델 평가가 왜곡되지 않도록 함

from collections import Counter
print('전체 분포 : ', Counter(y))
print('train 분포 : ', Counter(y_train))
print('test 분포 : ', Counter(y_test))

# 개별 모델 생성 (스케일링 - 표준화)
# make_pipeline을 이용해 전처리와 모델을 일체형으로 관리
logi = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver = 'lbfgs', max_iter = 1000, random_state = 12)
)
knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
tree = DecisionTreeClassifier(max_depth = 5, random_state = 12)

voting = VotingClassifier(
    estimators = [('LR', logi), ('KNN', knn), ('DT', tree)],
    voting = 'soft'
)

# 개별 모델 성능 확인
for clf in [logi, knn, tree]:
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    print(f'{clf.__class__.__name__} 정확도 : {accuracy_score(y_test, pred):.4f}')

voting.fit(x_train, y_train)
vpred = voting.predict(x_test)
print(f'voting 분류기 정확도 : {accuracy_score(y_test,vpred):.4f}')

# 옵션 : 교차 검증으로 안정성 확인
cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 12)
cv_score = cross_val_score(voting, x, y, cv = cv, scoring = 'accuracy')
print(f'voting 5겹 cv 평균 : {cv_score.mean():.4f} (+-) {cv_score.std():.4f}')
