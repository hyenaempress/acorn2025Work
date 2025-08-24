from scipy.stats import wilcoxon
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns

# 실습예제2)
# 여아신생아몸무게의평균검정수행 babyboom.csv
# 여아신생아의몸무게는평균이2800(g)으로알려져왔으나이보다더크다는주장이나왔다.
# 표본으로여아18명을뽑아체중을측정하였다고할때새로운주장이맞는지검정해보자

# 귀무가설(H0) : 여아 신생아의 몸무게는 2800g 이다.
# 대립가설(H1) : 여아 신생아의 몸무게는 2800g 이 아니다(크다).

data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/babyboom.csv')
# print(data2.head())
print(data2.describe)
print('-' * 100)
fdata = data2[data2['gender'] == 1]
print(fdata)
print('-' * 100)
print(len(fdata))
print('-' * 100)
print('mean : ',np.mean(fdata.weight), ' ', 'std : ',np.std(fdata.weight))
# mean :  3132.4444444444443   std :  613.7878951616051
# 2800g v.s 3132g 둘 사이에는 평균의 차이가 있는가?

# 정규성 검정 (하나의 집단일 때는 option)
print(stats.shapiro(fdata.iloc[:,2]))
print('-' * 100)
# mean :  3132.4444444444443   std :  613.7878951616051  정규성 위배
# 정규성 시각화
# 1) histogram으로 확인
sns.displot(fdata.weight, kde = True)
# plt.show()
plt.close()

# 2) Q-Q plot으로 확인
stats.probplot(fdata.weight, plot = plt) # 정규성을 확인하고 싶을 때 쓰면 좋은 probplot
# plt.show()
plt.close()
wilcox_resBaby = wilcoxon(fdata.weight - 2800) # 평균 2800과 비교
print('wilcox_resBaby : ', wilcox_resBaby)
print('-' * 100)
# WilcoxonResult(statistic=np.float64(37.0), pvalue=np.float64(0.03423309326171875))

resBaby = stats.ttest_1samp(fdata.weight, popmean = 2800)
print('statistic : %.5f, pvalue : %.5f' %resBaby)
print('-' * 100)
# statistic : 2.23319, pvalue : 0.03927 
# pvalue = 0.04 < 유의수준 0.05 이므로 귀무기각
# 즉, 여 신생아의 평균 체중은 2800g보다 증가하였다.

# [one-sample t 검정 : 문제1]  
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
#    305 280 296 313 287 240 259 266 318 280 325 295 315 278

# 귀무가설(H0) : 영사기에 사용되는 구형 백열전구의 수명은 300시간이다.
# 대립가설(H1) : 영사기에 사용되는 구형 백열전구의 수명은 300시간이 아니다.   

data2 = [305, 280, 296,313,287,240,259,266,318,280,325,295,315,278]
stat_data2 = stats.ttest_1samp(data2, popmean = 300)
print('전구 statistics : %.5f, pvalue : %.5f' %stat_data2)
print('-' * 100)
# 전구 statistics : -1.55644, pvalue : 0.14361
# pvalue = 0.14361 > 유의수준 0.05 이므로 귀무가설 채택
# 새로 개발된 전구의 수명시간은 300시간이 맞다.(H0)

# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
# 실습 파일 : one_sample.csv
# 참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")

# 귀무가설(H0) : 노트북의 평균 배터리 사용시간은 5.2시간이다.
# 대립가설(H1) : 노트북의 평균 배터리 사용시간은 5.2시간이 아니다.

lap_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')

print(lap_data)
print('-' * 100)
fdata3 = lap_data['time']
# print(fdata3)
fdata3 = fdata3.replace(['     ', ""],pd.NA)
fdata3 = fdata3.dropna()
fdata3 = pd.to_numeric(fdata3)
# fdata3 = int(fdata3)
print(fdata3)
print(len(fdata3))
print('-' * 100)

stat_data3 = stats.ttest_1samp(fdata3, popmean = 5.2)
print('노트북 수명 - statistic : %.5f, pvalue : %.5f'%stat_data3)
print('-' * 100)
# 노트북 수명 - statistic : 3.94606, pvalue : 0.00014
# pvalue값이 유의수준 0.05 보다 작으므로 귀무가설을 기각
# 노트북의 수명시간은 5.2시간이 아니다


# [one-sample t 검정 : 문제3] 
# https://www.price.go.kr/tprice/portal/main/main.do 에서 
# 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 
# 미용 요금을 얻도록 하자. 
# 정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.

# 귀무가설(H0) : 전국 평균 미용 요금은 15000원이다.
# 대립가설(H1) : 전국 평균 미용 요금은 15000원이 아니다.

data4 = pd.read_excel('개인서비스지역별_동향(2025-06월)819-11시9분.xlsx')
data4 = data4.T

data4 = data4.drop(['번호','품목'])

print(data4)
print('-' * 100)

data4 = data4.dropna()
print('drop 이후',data4)
print('-' * 100)

data4 = data4.iloc[:,[0]]
data4.columns = ['가격']
print('iloc 이후\n',data4)
print('-' * 100)

fdata4 = data4['가격']
fdata4 = pd.to_numeric(data4['가격'])
print(fdata4)
fdata4 = stats.ttest_1samp(fdata4, popmean = 15000)
print(fdata4)
# TtestResult(statistic=np.float64(6.67543428874073), 
# pvalue=np.float64(7.410669455004333e-06), df=np.int64(15))
# 피밸류는 0에 매우 근사한 값이므로 유의수준 0.05보다 아주 작다.
# 따라서 귀무가설을 기각한다.
# 전국 평균 미용 요금은 15000원이 아니다.