#분류모델 성능평가 관련

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

#로지스틱 리그레이션이란 
#분류모델입니다
#리니어 리그레이션은 

import numpy as np
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
print(X[:3])
print(y[:3])







