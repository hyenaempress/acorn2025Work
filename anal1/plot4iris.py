# iris (붓꽃) dataset : 꽃받침과 꽃잎의 너비와 길이로 꽃의 종류를 3종류로 구분해놓은 데이터 
# 각 그룹당 50개 , 총 150개 데이터 
#통계학 , 

import pandas as pd 
import matplotlib.pyplot as plt

iris_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/iris.csv')    
print(iris_data.head(3))
print(iris_data.tail(3))

#데이터 확인 
print(iris_data.info())

#데이터 타입 확인 
print(iris_data.dtypes)

#데이터 타입 변환 

#꽃받침의 길이와 꽃잎의 너비 

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length') 
plt.show() 
#색별로 표시해보자 

print(iris_data['Species'].unique())

#꽃받침의 길이와 꽃잎의 너비 

print(set(iris_data['Species']))

cols =[]
for s in iris_data['Species']:
    choice = 0
    if s == 'setosa':
        choice = 1
    elif s == 'versicolor':
        choice = 2
    else:
        choice = 3
    cols.append(choice)
print(cols)

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c=cols)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length') 
plt.title('Iris Data')
plt.show()


#데이터 분포와 산점도 그리기 

iris_col = iris_data.loc[:, 'Sepal.Length':'Petal.Width']
print(iris_col)

#판다스의 시각화 기능 넣어보기 
from pandas.plotting import scatter_matrix
scatter_matrix(iris_col, alpha=0.2, figsize=(6, 6), diagonal='kde') # 밀도 추청곡선을 넣었고, 대각선은 밀도 추청곡선을 넣었습니다. 
plt.show()

#이제 Seaborn 을 이용해서 그래프를 그려봅니다. 
import seaborn as sns
sns.pairplot(iris_data, hue='Species', height=2)
plt.show()

#러그 플롯 작은 선분으로 표시해주는 것입니다.
x = iris_data['Sepal.Length'].values
sns.rugplot(x)
plt.show()

#커널 밀도만 따로 보고 싶은 경우 
sns.kdeplot(x)
plt.show()  #데이터들이 비슷비슷해서 잘 안보입니다. 러그 플롯입니다 밀도를 부드러운 분포곡선 씨본을 사용하면 이렇게 볼 수 있다.
#아이리스 데이터셋 실습으로 이렇게 볼 수 있을 것 같습니다. 
#데이터를 보고 그때그때 찾아서 집어넣으면 됩니다. 





