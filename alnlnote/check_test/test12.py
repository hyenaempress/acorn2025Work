import pandas as pd
import numpy as np
from pandas import DataFrame

# 표준정규분포를 따르는 9×4 DataFrame 생성
df = DataFrame(np.random.randn(9, 4))

# 컬럼명을 "가격1, 가격2, 가격3, 가격4"로 지정
df.columns = ['가격1', '가격2', '가격3', '가격4']

# DataFrame 출력
print(df)

# 각 컬럼의 평균 계산 및 출력
print(df.mean())