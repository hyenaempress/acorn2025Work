# pandas로 파일 저장 및 읽기
import pandas as pd
import numpy as np

items = {
    'apple': {'count': 10, 'price': 1500},
    'orange': {'count': 4, 'price': 700},
    }

df = pd.DataFrame(items)
print(df)

# df.to_clipboard(index=False)  # 클립보드에 복사
# print(df.to_html())  # HTML로 변환
# print(df.to_json())  # JSON으로 변환

df.to_csv('result.csv', sep=',', index=False)
df.to_csv('result.csv', sep=',', index=False, header=False)  # 헤더 없이 저장

data = df.T
print(data)
data.to_csv('result.csv', sep=',', index=False) 

# 엑셀 관련
df2 = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Oscar'],
    'age': [25, 30, 22],
    'city': ['Seoul', 'Busan', 'Incheon']

})

print(df2)
df2.to_excel('result.xlsx', index=False, sheet_name='Sheet1')  # 엑셀 파일로 저장

# 읽기
exdf = pd.read_excel('result.xlsx')
print(exdf) 

#웹에서 모듈을 가져오는게 여러개 있다 

# pip install beautifulsoup4

