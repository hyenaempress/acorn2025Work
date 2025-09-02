# 앙상블 기법의 모델들 성능비교
# - 랜덤 포레스트
# - 그래디언트 부스팅
# - XGBoost
# - LightGBM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/titanic_data.csv') 
# print(df.info())  
df.drop(columns=['PassengerId','Name','Ticket'], inplace=True)
print(df.describe())
print(df.isnull().sum())

# null 처리
df['Age'].fillna(df['Age'].mean(), inplace=True)
# Age 열에 결측치(NaN)가 있을 때, 그 결측값을 Age 열의 평균값으로 모두 채워줍니다.
# 즉, 결측치가 있던 부분이 평균값으로 대체되어 데이터가 완성됩니다.
# inplace=True는 원본 데이터프레임(df)에 바로 적용한다는 뜻입니다.
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
print(df.info())

print('1. Sex : ', df['Sex'].value_counts())
# 데이터프레임 df의 Sex 열에 있는 각 값(예: 'male', 'female')이 몇 번씩 나오는지(빈도)를 세어서 출력합니다.
print('2. Cabin : ', df['Cabin'].value_counts())
print('3. Embarked : ', df['Embarked'].value_counts())
df['Cabin'] = df['Cabin'].str[:1]
print()

# Sex는 값의 종류가 2개('male', 'female')뿐이라서 Length가 생략됩니다.
# Cabin은 값의 종류가 148개로 많기 때문에, pandas가 자동으로 Length: 148을 출력해줍니다.
# 즉, 값의 종류가 많으면 Length(고유값 개수)가 표시되고,
# 값의 종류가 적으면 생략될 수 있습니다.

print(df.groupby(['Sex', 'Survived'])['Survived'].count())
print('여성 생존율 : ',233/ (233+81))
print('남성 생존율 : ',109/ (109+468))

sns.barplot(x='Sex', y='Survived', data=df, errorbar=('ci', 95))

# 성별 기준으로 Pclass별 생존 확률
plt.show()

# 나이별 기준으로 생존 확률
# 나이별 기준으로 생존 확률
def getAgeFunc(age):
    msg = ''
    if age <= -1:
        msg = 'unknown'
    elif age <= 5:
        msg = 'baby'
    elif age <= 18:
        msg = 'teenager'
    elif age <= 65:
        msg = 'adult'
    else:
        msg = 'elder'
    return msg

df['Age_category'] = df['Age'].apply(lambda a : getAgeFunc(a))
print(df.head(2))

sns.barplot(x='Age_category', y='Survived', hue='Sex', data=df, order=['unknown', 'baby', 'teenager', 'adult', 'elder'])
plt.show()
del df['Age_category']



from sklearn import preprocessing
def labelIncoder(datas):
    cols = ['Cabin', 'Sex','Embarked']
    for col in cols:
        le = preprocessing.LabelEncoder()
        # LabelEncoder는 범주형(문자) 데이터를 숫자(레이블)로 변환해주는 도구입니다.
        # 예를 들어, 'male', 'female'을 각각 0, 1로 바꿔줍니다.
        # LabelEncoder는 자동으로 알파벳 순서대로 0, 1, 2...로 변환합니다.
        # 예를 들어, 'female'이 0, 'male'이 1로 변환됩니다(알파벳 f < m)

        le.fit(datas[col])
        # le.fit(datas[col])로 해당 열의 고유값을 학습합니다.

        datas[col] = le.transform(datas[col])
        # datas[col] = le.transform(datas[col])로 각 값을 숫자(레이블)로 변환합니다. 
        #  - 모든 열에 대해 변환이 끝나면 변환된 데이터프레임을 반환합니다.
    return datas

df = labelIncoder(df)
print(df.head(3))
print(df['Cabin'].unique())
print(df['Sex'].unique())
print(df['Embarked'].unique())

print()

feature_df=df.drop(['Survived'],axis='columns')
label_df=df['Survived']
print(feature_df.head(3))
print(label_df.head(3))

x_train,x_test,y_train,y_test=train_test_split(feature_df,label_df,test_size=0.2,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
logmodel = LogisticRegression(solver='lbfgs',max_iter=500).fit(x_train,y_train)
rfmodel = RandomForestClassifier(criterion='entropy',n_estimators=500).fit(x_train,y_train)
demodel = DecisionTreeClassifier(criterion='entropy').fit(x_train,y_train)

logpred = logmodel.predict(x_test)
rfpred = rfmodel.predict(x_test)
depred = demodel.predict(x_test)

print('로지스틱 회귀 예측값 : ',logpred[:10])
print('랜덤 포레스트 예측값 : ',rfpred[:10])
print('결정 트리 예측값 : ',depred[:10])