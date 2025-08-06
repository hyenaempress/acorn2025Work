# Pandas의 DataFrame 관련 연습문제
import numpy as np
from pandas import Series
from pandas import DataFrame
import pandas as pd
import random

# 문제 1 - a)표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오.
print("\n\n문제 1 - a)표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오.")
# DataFrame() 생성자를 이용한 2차원 데이터 구조 생성
# 
# ▶ 입력(Input):
#   - np.random.randn(9, 4): 표준정규분포를 따르는 9행 4열 numpy 배열
#     * 평균=0, 표준편차=1인 정규분포에서 랜덤 샘플링
#     * 첫 번째 인자 9: 행의 개수
#     * 두 번째 인자 4: 열의 개수
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 9행 4열의 pandas DataFrame
#   - 컬럼명: 기본값 (0, 1, 2, 3)
#   - 인덱스: 기본값 (0, 1, 2, ..., 8)
q1_frame = DataFrame(np.random.randn(9, 4))
print("frame: \n", q1_frame)

# 문제 1 - b)a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오.
print("\n\n문제 1 - b)a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오.")
# DataFrame.columns 속성을 이용한 컬럼명 변경
# 
# ▶ 입력(Input):
#   - q1_frame.columns: 기존 DataFrame의 컬럼 인덱스 객체 (0, 1, 2, 3)
#   - ['No1', 'No2', 'No3', 'No4']: 새로운 컬럼명들의 리스트
# 
# ▶ 처리 과정:
#   - 기존 컬럼명을 새로운 이름으로 1:1 대응하여 변경
#   - 0 → No1, 1 → No2, 2 → No3, 3 → No4
# 
# ▶ 출력(Output):
#   - 원본 DataFrame의 컬럼명이 변경됨 (in-place 수정)
#   - 데이터는 그대로, 컬럼명만 변경된 상태
q1_frame.columns = ['No1', 'No2', 'No3', 'No4']
print("frame: \n", q1_frame)

# 문제 1 - c)각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용
print("\n\n문제 1 - c)각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용")
# DataFrame.mean() 함수를 이용한 평균값 계산
# 
# ▶ 입력(Input):
#   - q1_frame: 9행 4열의 DataFrame (No1, No2, No3, No4 컬럼)
#   - axis=0: 행 방향으로 계산 (세로 방향, 각 컬럼별 계산)
# 
# ▶ 처리 과정:
#   - 각 컬럼의 모든 행 값들을 더한 후 행 개수로 나눔
#   - axis=0: 컬럼별 평균 (axis=1이면 행별 평균)
# 
# ▶ 출력(Output):
#   - Series 객체: 각 컬럼명을 인덱스로 하고 평균값을 데이터로 하는 Series
#   - No1: 평균값, No2: 평균값, No3: 평균값, No4: 평균값
print("각 컬럼별 평균:\n", q1_frame.mean(axis=0))

# 문제 2
#     numbers
# a | 10
# b | 20
# c | 30
# d | 40
# 문제 2 - a)DataFrame으로 위와 같은 자료를 만드시오. column(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
print("\n\n문제 2 - a)DataFrame으로 위와 같은 자료를 만드시오. column(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.")
# DataFrame() 생성자를 이용한 1차원 데이터의 DataFrame 변환
# 
# ▶ 입력(Input):
#   - [10, 20, 30, 40]: 1차원 리스트 데이터 (4개 원소)
#   - columns=['numbers']: 컬럼명을 지정하는 파라미터 (리스트 형태)
#   - index=['a', 'b', 'c', 'd']: 행 인덱스명을 지정하는 파라미터 (리스트 형태)
# 
# ▶ 처리 과정:
#   - 1차원 리스트를 DataFrame의 단일 컬럼으로 변환
#   - 각 값과 인덱스를 1:1 대응: a-10, b-20, c-30, d-40
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 4행 1열 (컬럼명: numbers, 인덱스: a,b,c,d)
q2_frame = DataFrame([10, 20, 30, 40], columns=['numbers'], index=['a', 'b', 'c', 'd'])
print("q2_frame: \n", q2_frame)

# 문제 2 - b)c row의 값을 가져오시오.
print("\n\n문제 2 - b)c row의 값을 가져오시오.")
# DataFrame.loc[] 라벨 기반 인덱싱을 이용한 행 선택
# 
# ▶ 입력(Input):
#   - q2_frame: 4행 1열의 DataFrame (인덱스: a,b,c,d)
#   - 'c': 선택하고자 하는 행의 라벨(인덱스명)
# 
# ▶ 처리 과정:
#   - 라벨 'c'에 해당하는 행을 검색
#   - 해당 행의 모든 컬럼 값을 추출
# 
# ▶ 출력(Output):
#   - Series 객체: 인덱스 'c'에 해당하는 행 데이터
#   - 컬럼명을 인덱스로, 해당 행의 값들을 데이터로 하는 Series
print("q2_frame.loc['c']", q2_frame.loc['c'])

# 문제 2 - c)a, d row들의 값을 가져오시오.
print("\n\n문제 2 - c)a, d row들의 값을 가져오시오.")
# DataFrame.loc[] 다중 행 선택을 이용한 필터링
# 
# ▶ 입력(Input):
#   - q2_frame: 4행 1열의 DataFrame (인덱스: a,b,c,d)
#   - ['a', 'd']: 선택하고자 하는 여러 행의 라벨들을 담은 리스트
# 
# ▶ 처리 과정:
#   - 리스트에 포함된 각 라벨에 해당하는 행들을 검색
#   - 'a'와 'd' 라벨에 해당하는 행들만 추출
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 선택된 행들로 구성된 새로운 DataFrame
#   - 2행 1열 (인덱스: a,d / 컬럼: numbers)
print("q2_frame.loc['a', 'd']", q2_frame.loc[['a', 'd']])

# 문제 2 - d)numbers의 합을 구하시오.
# 컬럼 선택 및 집계함수를 이용한 합계 계산
# 
# ▶ 입력(Input):
#   - q2_frame: 4행 1열의 DataFrame
#   - ['numbers']: 선택하고자 하는 컬럼명
# 
# ▶ 처리 과정:
#   - q2_frame['numbers']: DataFrame에서 'numbers' 컬럼만 추출 → Series 객체 반환
#   - .sum(): Series의 모든 값들을 더하는 집계함수 수행
# 
# ▶ 출력(Output):
#   - 스칼라 값: 'numbers' 컬럼의 모든 값들의 합계 (10+20+30+40=100)
print("q2_frame['numbers'].sum():", q2_frame['numbers'].sum())

# 문제 2 - e)numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.
#     numbers
# a | 100
# b | 400
# c | 900
# d | 1600
print("\n\n문제 2 - e)numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.")
# DataFrame.multiply() 함수를 이용한 원소별 곱셈 연산
# 
# ▶ 입력(Input):
#   - q2_frame: 현재 DataFrame (numbers 컬럼: 10, 20, 30, 40)
#   - q2_frame: 자기 자신을 곱셈 피연산자로 사용
# 
# ▶ 처리 과정:
#   - multiply() 함수: 두 DataFrame의 같은 위치 원소끼리 곱셈 수행
#   - 자기 자신과 곱하므로 각 원소의 제곱 효과
#   - 10*10=100, 20*20=400, 30*30=900, 40*40=1600
# 
# ▶ 출력(Output):
#   - 원본 DataFrame이 수정됨 (in-place)
#   - numbers 컬럼의 각 값이 제곱된 새로운 DataFrame
q2_frame = q2_frame.multiply(q2_frame)
print("q2_frame:\n", q2_frame)

# 해결 방법 2:
# q2_frame = q2_frame * q2_frame
# print("q2_frame:\n", q2_frame)

# 문제 2 - f)float 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5 아래 결과가 나와야 함.
#     numbers   float
# a | 100     |  1.5
# b | 400     |  2.5
# c | 900     |  3.5
# d | 1600    |  4.5
print("\n\n문제 2 - f)float 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5 아래 결과가 나와야 함.")
# 임시 DataFrame 생성 후 컬럼 추가하는 방법
# 
# ▶ 입력(Input) - 임시 DataFrame 생성:
#   - [1.5, 2.5, 3.5, 4.5]: 실수 데이터 리스트
#   - columns=['float']: 새로 추가할 컬럼명
#   - index=['a', 'b', 'c', 'd']: 기존 DataFrame과 동일한 인덱스
# 
# ▶ 처리 과정:
#   - temp_frame: 새로운 데이터로 임시 DataFrame 생성
#   - temp_frame['float']: 임시 DataFrame에서 'float' 컬럼만 추출 → Series
#   - q2_frame['float'] = Series: 기존 DataFrame에 새 컬럼 추가
# 
# ▶ 출력(Output):
#   - 원본 DataFrame에 'float' 컬럼이 추가됨
#   - 2개 컬럼을 가진 DataFrame (numbers, float)
temp_frame = DataFrame([1.5, 2.5, 3.5, 4.5], columns=['float'], index=['a', 'b', 'c', 'd'])
q2_frame['float'] = temp_frame['float']
print("q2_frame:\n", q2_frame)

# 문제 2 - g)names 라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용
#       names
# d  |   길동
# a  |   오정
# b  |   팔계
# c  |   오공
print("\n\n문제 2 - g)names 라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용")
# 문자열 데이터로 새로운 컬럼 추가
# 
# ▶ 입력(Input):
#   - ['길동', '오정', '팔계', '오공']: 문자열 데이터 리스트
#   - columns=['names']: 새로 추가할 컬럼명
#   - index=['a', 'b', 'c', 'd']: 기존과 동일한 인덱스로 매칭
# 
# ▶ 처리 과정:
#   - temp_frame: 문자열 데이터로 임시 DataFrame 생성
#   - temp_frame['names']: 'names' 컬럼만 추출 → Series (문자열 타입)
#   - q2_frame에 새로운 컬럼으로 추가
# 
# ▶ 출력(Output):
#   - 원본 DataFrame에 'names' 컬럼이 추가됨
#   - 3개 컬럼을 가진 DataFrame (numbers, float, names)
temp_frame = DataFrame(['길동', '오정', '팔계', '오공'], columns=['names'], index=['a', 'b', 'c', 'd'])
q2_frame['names'] = temp_frame['names']
print("q2_frame:\n", q2_frame)

# 문제 3 - 1) 5 * 3 형태의 랜덤 정수형 DataFrame을 생성하시오.(범위: 1 이상 20 이하, 난수)
print("\n\n문제 3 - 1) 5 * 3 형태의 랜덤 정수형 DataFrame을 생성하시오.(범위: 1 이상 20 이하, 난수)")
# 정수형 난수 DataFrame 생성
# 
# ▶ 입력(Input):
#   - np.random.randint(): 정수형 난수 생성 함수
#     * 첫 번째 인자 1: 최솟값 (포함)
#     * 두 번째 인자 21: 최댓값 (제외, 실제로는 20까지)
#     * size=(5, 3): 배열의 형태 (5행 3열)
# 
# ▶ 처리 과정:
#   - 1~20 범위에서 랜덤한 정수 15개(5×3) 생성
#   - numpy 배열을 pandas DataFrame으로 변환
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 5행 3열의 정수형 데이터
#   - 기본 컬럼명: 0, 1, 2 / 기본 인덱스: 0, 1, 2, 3, 4
q3_frame = pd.DataFrame(np.random.randint(1, 21, size=(5, 3)))
print("q3_frame:\n", q3_frame)
# DataFrame.shape 속성을 이용한 차원 정보 확인
# 
# ▶ 출력(Output):
#   - 튜플: (행의 수, 열의 수) 형태로 DataFrame의 차원 정보 반환
print(q3_frame.shape)

# 문제 3 - 2) 생성된 DataFrame의 칼럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.
print("\n\n문제 3 - 2) 생성된 DataFrame의 칼럼 이름을 A, B, C로 설정하고, 행 인덱스를 r1, r2, r3, r4, r5로 설정하시오.")
# DataFrame의 컬럼명과 인덱스명 동시 변경
# 
# ▶ 입력(Input) - 컬럼명 변경:
#   - ['A', 'B', 'C']: 새로운 컬럼명들의 리스트
# 
# ▶ 처리 과정:
#   - q3_frame.columns: 기존 컬럼명 (0, 1, 2)을 새 이름으로 변경
#   - 0→A, 1→B, 2→C로 1:1 대응 변경
q3_frame.columns = ['A', 'B', 'C']

# ▶ 입력(Input) - 인덱스명 변경:
#   - ['r1', 'r2', 'r3', 'r4', 'r5']: 새로운 인덱스명들의 리스트
#   - pd.Index(): 리스트를 인덱스 객체로 변환하는 함수
# 
# ▶ 처리 과정:
#   - q3_frame.index: 기존 인덱스 (0,1,2,3,4)을 새 이름으로 변경
#   - 0→r1, 1→r2, 2→r3, 3→r4, 4→r5로 변경
# 
# ▶ 출력(Output):
#   - 컬럼명과 인덱스명이 모두 변경된 DataFrame
q3_frame.index = pd.Index(['r1', 'r2', 'r3', 'r4', 'r5'])
print("q3_frame:\n", q3_frame)

# 문제 3 - 3) A 컬럼의 값이 10보다 큰 행만 출력하시오.
print("\n\n문제 3 - 3) A 컬럼의 값이 10보다 큰 행만 출력하시오.")
# 불린 인덱싱(Boolean Indexing)을 이용한 조건부 필터링
# 
# ▶ 입력(Input):
#   - q3_frame['A']: DataFrame에서 'A' 컬럼만 추출 → Series 객체
#   - > 10: 비교 연산자로 각 원소와 10을 비교
# 
# ▶ 처리 과정:
#   - q3_frame['A'] > 10: A 컬럼의 각 값이 10보다 큰지 판별
#   - 결과: 불린 Series (True/False 배열) 생성
#   - q3_frame[불린Series]: True인 위치의 행들만 선택
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 조건을 만족하는 행들로 구성된 새로운 DataFrame
#   - A 컬럼 값이 10보다 큰 행들만 포함
print(q3_frame[q3_frame['A'] > 10])

# 문제 3 - 4) 새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.
print("\n\n문제 3 - 4) 새로 D라는 컬럼을 추가하여, A와 B의 합을 저장하시오.")
# DataFrame['새컬럼명'] = 연산식: 새로운 컬럼 추가 연산
# 
# ▶ 입력(Input):
#   - q3_frame['A']: DataFrame에서 'A' 컬럼만 추출 → Series 객체 반환
#   - q3_frame['B']: DataFrame에서 'B' 컬럼만 추출 → Series 객체 반환
#   - + 연산자: 두 Series 간의 원소별(element-wise) 덧셈 수행
# 
# ▶ 처리 과정:
#   - Series + Series → 각 행의 같은 인덱스끼리 덧셈 → 새로운 Series 생성
#   - 예: [10, 15, 8] + [5, 3, 12] = [15, 18, 20]
# 
# ▶ 출력(Output):
#   - q3_frame['D']: 기존 DataFrame에 새로운 'D' 컬럼이 추가됨
#   - 결과: 원본 DataFrame이 수정되어 A, B, C, D 4개 컬럼을 가지게 됨
q3_frame['D'] = q3_frame['A'] + q3_frame['B']
print("q3_frame:\n", q3_frame)

# 문제 3 - 5) 행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.
print("\n\n문제 3 - 5) 행 인덱스가 r3인 행을 제거하되, 원본 DataFrame이 실제로 바뀌도록 하시오.")
# DataFrame.drop() 함수를 이용한 행 제거
# 
# ▶ 입력(Input):
#   - q3_frame: 현재 DataFrame (r1, r2, r3, r4, r5, r6 인덱스)
#   - 'r3': 제거하고자 하는 행의 인덱스명
#   - inplace=True: 원본 수정 옵션
# 
# ▶ 처리 과정:
#   - 'r3' 인덱스를 가진 행을 찾아서 제거
#   - inplace=True: 새로운 DataFrame을 반환하지 않고 원본을 직접 수정
#   - inplace=False(기본값): 새로운 DataFrame을 반환하고 원본은 유지
# 
# ▶ 출력(Output):
#   - 원본 DataFrame에서 r3 행이 제거됨 (in-place 수정)
#   - 반환값: None (inplace=True이므로)
q3_frame.drop('r3', inplace=True)
print("q3_frame:\n", q3_frame)

# 문제 3 - 6) 아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.
#       A       B       C       D
#r6     15      10      2       A+B
print("\n\n문제 3 - 6) 아래와 같은 정보를 가진 새로운 행(r6)을 DataFrame 끝에 추가하시오.")
# DataFrame.loc['새인덱스명'] = [값들]: 새로운 행을 추가하는 방법
# 
# ▶ 입력(Input):
#   - q3_frame: 현재 DataFrame (r3이 제거된 상태: r1, r2, r4, r5)
#   - 'r6': 새로 추가할 행의 인덱스명 (문자열)
#   - [15, 10, 2, 15+10]: 각 컬럼(A, B, C, D)에 대응하는 값들의 리스트
# 
# ▶ 처리 과정:
#   - loc[] 접근자를 이용해 존재하지 않는 인덱스('r6')에 값 할당
#   - [15, 10, 2, 25]: A=15, B=10, C=2, D=25로 새로운 행 생성
#   - 15+10: D 컬럼에는 A+B 합계값을 계산하여 저장
# 
# ▶ 출력(Output):
#   - 원본 DataFrame에 'r6' 인덱스를 가진 새로운 행이 추가됨
#   - 결과: r1, r2, r4, r5, r6 총 5개 행을 가진 DataFrame
q3_frame.loc['r6'] = [15, 10, 2, 15+10]
print("q3_frame:\n", q3_frame)

# 문제 4) 다음과 같은 재고 정보를 가지고 있는 딕셔너리 Data가 있다고 하자.
# data = {
#     'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],
#     'price':   [12000,     25000,      150000,    900000],
#     'stock':   [  10,         5,          2,          3 ]
# }
print("\n\n문제 4) 다음과 같은 재고 정보를 가지고 있는 딕셔너리 Data가 있다고 하자.")
# 딕셔너리 형태로 데이터 정의
# 각 키는 컬럼명이 되고, 각 값(리스트)은 해당 컬럼의 데이터가 됨
data = {
    'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop'],    # 상품명 리스트
    'price':   [12000,     25000,      150000,    900000],   # 가격 리스트
    'stock':   [  10,         5,          2,          3 ]    # 재고 리스트
}
print("data:\n", data)

# 문제 4 - 1) 위 딕셔너리로부터 DataFrame을 생성하시오. 단, 행 인덱스는 p1, p2, p3, p4가 되도록 하시오.
print("문제 4 - 1) 위 딕셔너리로부터 DataFrame을 생성하시오. 단, 행 인덱스는 p1, p2, p3, p4가 되도록 하시오.")
# DataFrame() 생성자에 딕셔너리 전달하여 DataFrame 생성
# 
# ▶ 입력(Input):
#   - data: 딕셔너리 객체 (키: 컬럼명, 값: 데이터 리스트)
#     * 'product': ['Mouse', 'Keyboard', 'Monitor', 'Laptop']
#     * 'price': [12000, 25000, 150000, 900000]
#     * 'stock': [10, 5, 2, 3]
#   - index=['p1', 'p2', 'p3', 'p4']: 행 인덱스 지정
# 
# ▶ 처리 과정:
#   - 딕셔너리의 각 키가 DataFrame의 컬럼명이 됨
#   - 각 키의 값(리스트)이 해당 컬럼의 데이터가 됨
#   - 행 인덱스를 기본값(0,1,2,3) 대신 지정된 값(p1,p2,p3,p4)으로 설정
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 4행 3열 (인덱스: p1~p4, 컬럼: product, price, stock)
q4_frame = DataFrame(data, index = ['p1', 'p2', 'p3', 'p4'])
print("q4_frame: \n", q4_frame)

# 문제 4 - 2) price와 stock을 이용하여 'total'이라는 새로운 컬럼을 추가하고, 값은 'price x stock'이 되도록 하시오.
print("\n\n문제 4 - 2) price와 stock을 이용하여 'total'이라는 새로운 컬럼을 추가하고, 값은 'price x stock'이 되도록 하시오.")
# DataFrame 컬럼 간 연산을 통한 새로운 컬럼 생성
# 
# ▶ 입력(Input):
#   - q4_frame['price']: DataFrame에서 'price' 컬럼만 추출 → Series 객체
#     * [12000, 25000, 150000, 900000]
#   - q4_frame['stock']: DataFrame에서 'stock' 컬럼만 추출 → Series 객체
#     * [10, 5, 2, 3]
#   - * 연산자: 두 Series 간의 원소별(element-wise) 곱셈 수행
# 
# ▶ 처리 과정:
#   - Series * Series → 각 행의 같은 인덱스끼리 곱셈 → 새로운 Series 생성
#   - p1: 12000 * 10 = 120000, p2: 25000 * 5 = 125000
#   - p3: 150000 * 2 = 300000, p4: 900000 * 3 = 2700000
# 
# ▶ 출력(Output):
#   - q4_frame['total']: 기존 DataFrame에 새로운 'total' 컬럼 추가
#   - 결과: 4개 컬럼을 가진 DataFrame (product, price, stock, total)
q4_frame['total'] = q4_frame['price'] * q4_frame['stock']
print("q4_frame: \n", q4_frame)

# 문제 4 - 3) 컬럼 이름을 다음과 같이 변경하시오. 원본 갱신
# - product → 상품명,  price → 가격,  stock → 재고,  total → 총가격
print("\n\n문제 4 - 3) 컬럼 이름을 다음과 같이 변경하시오. 원본 갱신")
# DataFrame.rename() 함수를 이용한 컬럼명 변경
# 
# ▶ 입력(Input):
#   - q4_frame: 현재 DataFrame (컬럼: product, price, stock, total)
#   - columns={기존명: 새이름}: 변경할 컬럼명들을 담은 딕셔너리
#     * 'product': '상품명', 'price': '가격'
#     * 'stock': '재고', 'total': '총가격'
#   - inplace=True: 원본 수정 옵션
# 
# ▶ 처리 과정:
#   - 딕셔너리의 각 키-값 쌍에 따라 컬럼명을 1:1 대응하여 변경
#   - product→상품명, price→가격, stock→재고, total→총가격
#   - inplace=True: 새로운 DataFrame을 반환하지 않고 원본을 직접 수정
# 
# ▶ 출력(Output):
#   - 원본 DataFrame의 컬럼명이 한글로 변경됨 (in-place 수정)
#   - 반환값: None (inplace=True이므로)
q4_frame.rename(
        columns={
            'product': '상품명',    # product 컬럼을 '상품명'으로 변경
            'price': '가격',       # price 컬럼을 '가격'으로 변경
            'stock': '재고',       # stock 컬럼을 '재고'로 변경
            'total': '총가격'      # total 컬럼을 '총가격'으로 변경
    },
    inplace=True
)
print("q4_frame: \n", q4_frame)

# 문제 4 - 4) 재고(재고 칼럼)가 3 이하인 행의 정보를 추출하시오.
print("\n\n문제 4 - 4) 재고(재고 칼럼)가 3 이하인 행의 정보를 추출하시오.")
# 불린 인덱싱을 이용한 조건부 데이터 필터링
# 
# ▶ 입력(Input):
#   - q4_frame['재고']: DataFrame에서 '재고' 컬럼만 추출 → Series 객체
#   - <= 3: 비교 연산자로 각 원소와 3을 비교
# 
# ▶ 처리 과정:
#   - q4_frame['재고'] <= 3: '재고' 컬럼의 각 값이 3 이하인지 판별
#   - 결과: 불린 Series (True/False 배열) 생성
#     * p1: 10 <= 3 → False, p2: 5 <= 3 → False
#     * p3: 2 <= 3 → True, p4: 3 <= 3 → True
#   - q4_frame[불린Series]: True인 위치의 행들만 선택
# 
# ▶ 출력(Output):
#   - DataFrame 객체: 조건을 만족하는 행들로 구성된 새로운 DataFrame
#   - 재고가 3 이하인 행들만 포함 (p3: Monitor, p4: Laptop)
print(q4_frame[q4_frame['재고'] <= 3])

# 문제 4 - 5) 인덱스가 p2인 행을 추출하는 두 가지 방법(loc, iloc)을 코드로 작성하시오.
print("\n\n문제 4 - 5) 인덱스가 p2인 행을 추출하는 두 가지 방법(loc, iloc)을 코드로 작성하시오.")
# 라벨 기반 인덱싱 vs 위치 기반 인덱싱 비교
# 
# ▶ 방법 1: DataFrame.loc[] - 라벨 기반 인덱싱
# ▶ 입력(Input):
#   - q4_frame: 현재 DataFrame (인덱스: p1, p2, p3, p4)
#   - 'p2': 행 인덱스 라벨(이름)
# ▶ 출력(Output):
#   - Series 객체: 'p2' 인덱스에 해당하는 행의 모든 데이터
print("q4_frame.loc['p2']", q4_frame.loc['p2'])

# ▶ 방법 2: DataFrame.iloc[] - 위치 기반 인덱싱
# ▶ 입력(Input):
#   - q4_frame: 현재 DataFrame
#   - 1: 정수 위치 (0부터 시작, p2는 두 번째 행이므로 인덱스 1)
# ▶ 처리 과정:
#   - p1(0번째), p2(1번째), p3(2번째), p4(3번째)
# ▶ 출력(Output):
#   - Series 객체: 1번째 위치에 해당하는 행의 모든 데이터
print("q4_frame.iloc[2]", q4_frame.iloc[1])

# 문제 4 - 6) 인덱스가 p3인 행을 삭제한 뒤, 그 결과를 확인하시오.(원본이 실제로 바뀌지 않도록, 즉 drop()의 기본 동작으로)
print("\n\n문제 4 - 6) 인덱스가 p3인 행을 삭제한 뒤, 그 결과를 확인하시오.(원본이 실제로 바뀌지 않도록, 즉 drop()의 기본 동작으로)")
# DataFrame.drop() 함수의 기본 동작 (원본 보존)
# 
# ▶ 입력(Input):
#   - q4_frame: 현재 DataFrame (인덱스: p1, p2, p3, p4)
#   - 'p3': 제거하고자 하는 행의 인덱스명
#   - inplace=False: 원본 보존 옵션 (기본값)
# 
# ▶ 처리 과정:
#   - 'p3' 인덱스를 가진 행을 찾아서 제거
#   - inplace=False: 원본은 그대로 두고 새로운 DataFrame을 생성하여 반환
#   - 원본 q4_frame은 변경되지 않음
# 
# ▶ 출력(Output):
#   - 새로운 DataFrame 객체: p3 행이 제거된 복사본 (p1, p2, p4만 포함)
#   - 원본 q4_frame은 여전히 p1, p2, p3, p4를 모두 포함
print("q4_frame.drop('p3', inplace=False):", q4_frame.drop('p3', inplace=False))

# 문제 4 - 7) 위 DataFrame에 아래와 같은 행(p5)을 추가하시오.
#         상품명       가격       재고     총가격
#  p5     USB메모리    15000      10      가격*재고
print("\n\n문제 4 - 7) 위 DataFrame에 아래와 같은 행(p5)을 추가하시오.")
# DataFrame.loc[]을 이용한 새로운 행 추가
# 
# ▶ 입력(Input):
#   - q4_frame: 현재 DataFrame (인덱스: p1, p2, p3, p4)
#   - 'p5': 새로 추가할 행의 인덱스명 (문자열)
#   - ['USB메모리', 15000, 10, 15000*10]: 각 컬럼에 대응하는 값들의 리스트
#     * 상품명: 'USB메모리' (문자열)
#     * 가격: 15000 (정수)
#     * 재고: 10 (정수)
#     * 총가격: 150000 (계산된 값)
# 
# ▶ 처리 과정:
#   - loc[] 접근자를 이용해 존재하지 않는 인덱스('p5')에 값 할당
#   - 각 값이 해당 컬럼(상품명, 가격, 재고, 총가격) 순서대로 매핑됨
#   - 15000*10: 가격과 재고를 곱한 총가격을 실시간 계산
# 
# ▶ 출력(Output):
#   - 원본 DataFrame에 'p5' 인덱스를 가진 새로운 행이 추가됨
#   - 결과: p1, p2, p3, p4, p5 총 5개 행을 가진 DataFrame
q4_frame.loc['p5'] = ['USB메모리', 15000, 10, 15000*10]
print("q4_frame: \n", q4_frame)