#[one-sample t 검정 : 문제1]  
#영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
#한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
#연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
#한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
 #  305 280 296 313 287 240 259 266 318 280 325 295 315 278

import pandas as pd
import numpy as np
import scipy.stats as stats

data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print(data)
print(len(data))        #데이터 개수
print(np.mean(data))    #평균
print(np.std(data))     #표준편차
print(stats.ttest_1samp(data, 250))  #t-test 검정
print(stats.ttest_1samp(data, 250).pvalue)  #pvalue가 0.05보다 작으므로 귀무가설 기각   

#[one-sample t 검정 : 문제2] 
#국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
#실습 파일 : one_sample.csv https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv
#참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")

import pandas as pd
import numpy as np
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
print(data)
print(data.info())
print(data.describe())

#[one-sample t 검정 : 문제3] 
#https://www.price.go.kr/tprice/portal/main/main.do 에서 
#메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
#정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.

#미용 요금 데이터 가져오기
import pandas as pd
import numpy as np
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
print(data)
print(data.info())  #데이터 타입 확인
print(data.describe())  #기초 통계량 확인

#미용 요금 데이터 전처리
data['time'] = data['time'].str.replace('     ', '')
print(data)

#미용 요금 데이터 분석
print(stats.ttest_1samp(data['time'], 15000))   #t-test 검정
print(stats.ttest_1samp(data['time'], 15000).pvalue)  #pvalue가 0.05보다 작으므로 귀무가설 기각

#미용 요금 데이터 시각화
import matplotlib.pyplot as plt
plt.hist(data['time'])
plt.show()

#미용 요금 데이터 시각화
plt.boxplot(data['time'])   #boxplot 그리기 
plt.show()

#미용 요금 데이터 시각화
plt.hist(data['time'])   #히스토그램 그리기
plt.show()

#미용 요금 데이터 시각화
plt.boxplot(data['time'])   #boxplot 그리기
plt.show()      