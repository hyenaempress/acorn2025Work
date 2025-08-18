import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt


# 데이터 로드
data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/cleanDescriptive.csv")

# 귀무가설(H0): 부모 학력 수준과 자녀의 대학 진학 여부는 서로 관련이 없다.(독립)
# 대립가설(H1): 부모 학력 수준과 자녀의 대학 진학 여부는 서로 관련이 있다.(독립이 아니다)

print(data)

print(data['level2'].unique()) # leve = [ 1.  2. nan  3.]   1=고졸, 2=대졸 3=대학원졸  / level2 =['고졸' '대졸' nan '대학원졸']
print(data['pass2'].unique()) # pass = [ 2.  1. nan]   2=실패 1=합격  / pass2 =  실패  합격

# 부모학력 수준과 자녀의 진학여부 교차표
ctab = pd.crosstab(index=data['level2'], columns=data['pass2'])

print(ctab)
ctab.index = ["고졸", "대졸", "대학원졸"]
ctab.columns = ["합격" , "실패"]

# print(ctab)

chi2, p , dof, _ = stats.chi2_contingency(ctab)
msg = "test statics:{}, p-value:{}, df:{}"
format(msg.format(chi2, p , dof))
print(msg.format(chi2,p,dof))
# test statics:2.7669512025956684, p-value:0.25070568406521365, df:2
"""
결론 : p-value(0.25070)  > 0.05 이므로 귀무가설 채택.
부모 학력 수준과 자녀의 대학 진학 여부는 서로 관련이 없다. (독립)

"""

# 추가 그래프 
plt.rc('font', family='malgun gothic')
sns.heatmap(ctab, annot=True, fmt='d', cmap='Blues')   
# data-> ctab는 숫자형 데이터(빈도수)만 포함하고 있어 heatmap이 정상적으로 작동되게함.
plt.title('부모 학력 수준과 자녀 진학 여부')
plt.xlabel('자녀 진학 여부 (pass)')
plt.ylabel('부모 학력 수준 (level)')
plt.show()
