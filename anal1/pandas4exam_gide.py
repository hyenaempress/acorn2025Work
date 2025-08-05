import pandas as pd
import numpy as np

print("=" * 50)
print("Pandas DataFrame 연습문제")
print("=" * 50)

# =============================================================================
# 문제 1: 표준정규분포 DataFrame
# =============================================================================
print("\n문제 1: 표준정규분포 DataFrame")
print("-" * 30)

# a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame 생성
df1 = pd.DataFrame(np.random.randn(9, 4))
print("a) 9x4 DataFrame 생성:")
print(df1)

# b) 컬럼 이름을 No1, No2, No3, No4로 지정
df1.columns = ['No1', 'No2', 'No3', 'No4']
print("\nb) 컬럼명 지정 후:")
print(df1)

# c) 각 컬럼의 평균 구하기
print("\nc) 각 컬럼의 평균:")
print(df1.mean(axis=0))

# =============================================================================
# 문제 2: 기본 DataFrame 조작
# =============================================================================
print("\n\n문제 2: 기본 DataFrame 조작")
print("-" * 30)

# a) DataFrame 생성
data2 = {'numbers': [10, 20, 30, 40]}
index2 = ['a', 'b', 'c', 'd']
df2 = pd.DataFrame(data2, index=index2)
print("a) DataFrame 생성:")
print(df2)

# b) c row의 값 가져오기
print("\nb) c row의 값:")
print(df2.loc['c'])

# c) a, d row들의 값 가져오기
print("\nc) a, d row들의 값:")
print(df2.loc[['a', 'd']])

# d) numbers의 합 구하기
print("\nd) numbers의 합:")
print(df2['numbers'].sum())

# e) numbers의 값들을 각각 제곱
df2['numbers'] = df2['numbers'] ** 2
print("\ne) numbers 제곱 후:")
print(df2)

# 원래 값으로 복원 (다음 문제를 위해)
df2['numbers'] = [10, 20, 30, 40]

# f) floats 컬럼 추가
df2['floats'] = [1.5, 2.5, 3.5, 4.5]
print("\nf) floats 컬럼 추가:")
print(df2)

# g) names 컬럼 추가 (Series 사용)
names_series = pd.Series(['오정', '팔계', '오공', '길동'], index=['a', 'b', 'c', 'd'])
df2['names'] = names_series
print("\ng) names 컬럼 추가:")
print(df2)

# =============================================================================
# 문제 3: 랜덤 정수형 DataFrame
# =============================================================================
print("\n\n문제 3: 랜덤 정수형 DataFrame")
print("-" * 30)

# 1) 5 x 3 형태의 랜덤 정수형 DataFrame 생성 (1~20 범위)
np.random.seed(42)  # 재현 가능한 결과를 위해
df3 = pd.DataFrame(np.random.randint(1, 21, size=(5, 3)))
print("1) 5x3 랜덤 정수형 DataFrame:")
print(df3)

# 2) 컬럼명과 행 인덱스 설정
df3.columns = ['A', 'B', 'C']
df3.index = ['r1', 'r2', 'r3', 'r4', 'r5']
print("\n2) 컬럼명과 인덱스 설정:")
print(df3)

# 3) A 컬럼의 값이 10보다 큰 행만 출력
print("\n3) A 컬럼의 값이 10보다 큰 행:")
print(df3[df3['A'] > 10])

# 4) D 컬럼 추가 (A + B)
df3['D'] = df3['A'] + df3['B']
print("\n4) D 컬럼 추가 (A + B):")
print(df3)

# 5) r3 행 제거 (원본 변경)
df3.drop('r3', inplace=True)
print("\n5) r3 행 제거 후:")
print(df3)

# 6) r6 행 추가
new_row = pd.DataFrame({'A': [15], 'B': [10], 'C': [2], 'D': [25]}, index=['r6'])
df3 = pd.concat([df3, new_row])
print("\n6) r6 행 추가:")
print(df3)

# =============================================================================
# 문제 4: 재고 정보 DataFrame
# =============================================================================
print("\n\n문제 4: 재고 정보 DataFrame")
print("-" * 30)

# 1) 딕셔너리로부터 DataFrame 생성
data = {
    'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],
    'price':   [12000,   25000,     150000,    900000],
    'stock':   [10,      5,         2,         3]
}
df4 = pd.DataFrame(data, index=['p1', 'p2', 'p3', 'p4'])
print("1) 재고 정보 DataFrame:")
print(df4)

# 2) total 컬럼 추가 (price x stock)
df4['total'] = df4['price'] * df4['stock']
print("\n2) total 컬럼 추가:")
print(df4)

# 3) 컬럼명 변경
df4.rename(columns={
    'product': '상품명',
    'price': '가격',
    'stock': '재고',
    'total': '총가격'
}, inplace=True)
print("\n3) 컬럼명 변경:")
print(df4)

# 4) 재고가 3 이하인 행 추출
print("\n4) 재고가 3 이하인 행:")
print(df4[df4['재고'] <= 3])

# 5) p2 행 추출하는 두 가지 방법
print("\n5) p2 행 추출:")
print("loc 사용:")
print(df4.loc['p2'])
print("\niloc 사용:")
print(df4.iloc[1])  # p2는 인덱스 1번

# 6) p3 행 삭제 (원본 변경 없이)
df4_dropped = df4.drop('p3')
print("\n6) p3 행 삭제 후 (원본 변경 없음):")
print(df4_dropped)
print("\n원본 확인:")
print(df4)

# 7) p5 행 추가
new_row_p5 = pd.DataFrame({
    '상품명': ['USB메모리'],
    '가격': [15000],
    '재고': [10],
    '총가격': [15000 * 10]
}, index=['p5'])
df4_with_p5 = pd.concat([df4, new_row_p5])
print("\n7) p5 행 추가:")
print(df4_with_p5)

print("\n" + "=" * 50)
print("모든 연습문제 완료!")
print("=" * 50)