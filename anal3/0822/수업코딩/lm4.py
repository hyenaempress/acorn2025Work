# 방법4: linegress model O
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 값 예측
score_iq = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())

x = score_iq.iq
y = score_iq.score

# 상관 계수 확인
print(np.corrcoef(x, y)[0, 1]) # 0.88222
print(score_iq.corr())

# plt.scatter(x, y)
# plt.show()

model = stats.linregress(x, y)
print(model)
# LinregressResult(
#     slope=np.float64(0.6514309527270075), 
#     intercept=np.float64(-2.8564471221974657), 
#     rvalue=np.float64(0.8822203446134699), 
#     pvalue=np.float64(2.8476895206683644e-50), 
#     stderr=np.float64(0.028577934409305443), 
#     intercept_stderr=np.float64(3.546211918048538)
#     )
print("기울기:", model.slope) # 기울기 0.651
print("절편 :", model.intercept) # -2.856
print("R² - 결정계수 :", model.rvalue) # 0.882
print("p-value:", model.pvalue) # p-value 2.8476895206683644e-50 이므로 현재 모델은 유의하다.(독립변수와 종속변수는 인과관계가 있다.)
print("표준오차: ", model.stderr) # 0.028 -> 작을수록 좋다.
# ŷ = 0.651 * x - 2.856
plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept, color='red')

# 점수 예측
print('점수 예측: ', model.slope * 80 + model.intercept)
print('점수 예측: ', model.slope * 100 + model.intercept)
# predict
print('점수 예측: ', np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))

print()
newdf = pd.DataFrame({'iq': [55, 66, 77, 88, 150]})
print('점수 예측: \n', np.polyval([model.slope, model.intercept], newdf))
