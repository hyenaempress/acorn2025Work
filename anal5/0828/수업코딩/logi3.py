# Logistic Regression 클래스: 다항분류 가능
# 활성화 함수: softmax

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris() # 붓꽃 데이터셋
# print(iris.DESCR)
print(iris.keys())
print(iris.target)
x = iris.data[:, [3]] # petal.length
# print(x)
y = (iris.target == 2).astype(np.int32)
# print(y[:3])
# print(type(y))

log_reg = LogisticRegression().fit(x, y) # solver : lbfgs (softmax 사용)
print(log_reg)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1) # 새로운 예측 값을 얻기 위해 독립 변수생성
print(x_new)
y_proba = log_reg.predict_proba(x_new)
print(y_proba)

plt.plot(x_new, y_proba[:, 1], 'r-', label = 'virginica')
plt.plot(x_new, y_proba[:, 0], 'b--', label = 'not virginica')
plt.xlabel('petal width')
plt.legend()
plt.show()
plt.close()

