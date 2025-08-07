#seaborn :matplotlib의 기능 보강용 모듈
import matplotlib.pyplot as plt
import seaborn as sns 

titanic=sns.load_dataset('titanic')
# print(titanic.info()) 

# sns.kdeplot(y='age',data=titanic,palette='Paired')
# plt.show()

# sns.relplot(x='who', y='age', data=titanic)
# plt.show()

# sns.countplot(x='class', data=titanic)
# plt.show()

# t_pivot = titanic.pivot_table(index='class', columns='sex', aggfunc='size')
# print(t_pivot)

sns.heatmap(t_pivot, cmap=sns.light_palette('gray', as_cmap=True), annot=True, fmt='d')
plt.show()