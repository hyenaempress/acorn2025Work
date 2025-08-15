# 일원 카이제곱 검정 : 변인이 1개
# 적합도(선호도) 검정
# 실험을 통해 얻은 관찰값들이 어떤 이론적 분포를 따르고 있는지 확인하는 검정
# 꽃 색깔의 표현 분리 비율이 3:1이 맞는가?

# <적합도 검정실습>
# 주사위를 60 회 던져서 나온 관측도수 / 기대도수가 아래와 같이 나온 경우에 이 주사위는 적합한 주사위가 맞는가를 일원카이제곱 검정
# 으로 분석하자.

# 가설 세우기
# - 대립가설: 기대치와 관찰치는 차이가 있다. (주사위는 공정하지 않다.)
# - 귀무가설: 기대치와 관찰치는 동일하다. (주사위는 공정하다.)

import pandas as pd 
import scipy.stats as stats # 카이 제곱 검증에는 다양한 라이브러리를 사용할 수 있다.
import matplotlib.pyplot as plt
import numpy as np

data = [4, 6, 17, 16, 8, 9] 
# 관측 값
exp = [10, 10, 10, 10, 10, 10] # 기대 값

print(stats.chisquare(data))

#Power_divergenceResult(statistic=np.float64(14.200000000000001), pvalue=np.float64(0.014387678176921308))
# 카이제곱: 14.2, p-value: 0.0144
# 결론: 유의수준(0.05 < p-value) 귀무 기각
# 주사위는 게임에 적합하지 않다.
# 관측값은 우연히 발생한 것이 아니라, 어떠한 원인에 의해 발생한 얻어진 값이다.
print(stats.chisquare(data, exp))
result = stats.chisquare(data, exp)
print('chi2: ', result[0])
print('p-value: ', result[1])

# <선호도 분석 실습>
# 5개의 스포츠 음료에 대한 선호도에 차이가 있는지 검정하기
sports_drinks = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv')
print(sports_drinks)

print(stats.chisquare(sports_drinks['관측도수']))
# Power_divergenceResult(statistic=np.float64(20.488188976377952), pvalue=np.float64(0.00039991784008227264))
# 결과: p-value: 0.0003 < 0.05 귀무 기각

# 시각화: 어떤 음료가 기대보다 많이 선호되는지 확인

# 기대 도수
total = sports_drinks['관측도수'].sum()
expected = [total / len(sports_drinks) ] * len(sports_drinks)
print('expected: ', expected)

x = np.arange(len(sports_drinks))
width = 0.35 # 막대 너비
plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(9, 5))
plt.bar(x = (x - width / 2), height = sports_drinks['관측도수'], width=width, label='관측도수')
plt.bar(x = (x - width / 2), height = expected, width=width, label='기대도수', alpha = 0.6)
plt.xticks(x, list(sports_drinks['음료종류']))
plt.xlabel("음료 종류")
plt.ylabel("도수")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# 그래프의 카이제곱 검정 결과를 바탕으로 어떤 음료가 더 인기 있는지 구체적으로 분석
# 총합과 기대도수 이미 구함
# 차이 계산: 
sports_drinks['기대도수'] = expected
sports_drinks['차이(관측-기대)'] = sports_drinks['관측도수'] - sports_drinks['기대도수']
sports_drinks['차이비율(%)'] = round(sports_drinks['차이(관측-기대)'] / expected * 100, 2)
print(sports_drinks.head(3))
sports_drinks.sort_values(by='차이(관측-기대)', ascending=False, inplace=True) # 내림차순 정렬, 원본 데이터 수정
sports_drinks.reset_index(drop=True, inplace=True)
print(sports_drinks.head(3))