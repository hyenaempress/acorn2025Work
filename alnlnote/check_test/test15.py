import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (한글 깨짐 방지)
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False


#CSV 파일 읽기
df = pd.read_csv('sales_data.csv', encoding='utf-8')
#print(df.head())

# 2. 피봇 테이블을 위해서 집계 합 수량 구하기 
pivot_result = df.pivot_table(
    values='판매수량',    # 집계할 값
    index='날짜',         # 행에 배치할 변수
    columns='제품',       # 열에 배치할 변수
    aggfunc='sum'         # 집계 함수 (합계)
)

#print(pivot_result)

plt.figure(figsize=(10, 6))
plt.plot(pivot_result.index, pivot_result['노트북'], 'o-', label='노트북', color='blue')
plt.plot(pivot_result.index, pivot_result['마우스'], 'o-', label='마우스', color='orange')

plt.xlabel('날짜')
plt.ylabel('판매수량')
plt.legend(title='제품')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=0)
plt.show()