#seaborn : 은 맥플로리브의 기능보강용 
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Sea본은 데이터 몇개를 해주는데 

#데이터 준비 
titanic = sns.load_dataset('titanic') #타이타닉 데이터를 가져옵니다
print(titanic.info()) 

#타이타닉으로 분류 예측 모델 만들 수 있습니다.

sns.boxplot(y='age', data=titanic, palette='Paired') #타이타닉에서 age 만 빼네서 박스 플랏을 만든다 
plt.show()

#sns.distplot(titanic['age']) #밀도 그래프 
sns.kdeplot(titanic['age']) 
plt.show() 

#카테고리 범주 
sns.relplot(x='who', y= 'age', data=titanic)
plt.show()

sns.catplot(x='calss', y='age', data=titanic)
plt.show()

#피봇테이블 

t_pivot = titanic.pivot_table(index='class', columns='sex', aggfunc='size')
print(t_pivot)


#히트맵 
sns.heatmap(t_pivot, annot=True, fmt='d', cmap=sns.light_palette('gray', as_cmap=True))
plt.show()

#히트맵 옵션 
sns.heatmap(t_pivot, annot=True, fmt='d', cmap=sns.color_palette('RdBu', n_colors=10), center=0)
plt.show()

#데이터 분석하는 사람이 꼭 만나는 아이리스 데이터
# 엠리스트 데이터 

