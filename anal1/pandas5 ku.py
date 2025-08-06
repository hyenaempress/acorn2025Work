# file i/o
import pandas as pd

df = pd.read_csv('./code/0806/ex1.csv', sep = ' ')
print(df, type(df))
print(df.info())

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df, type(df))
print(df.info())

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)
print(df, type(df))
print(df.info())

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a', 'b', 'c', 'd', 'msg'], skiprows=1)
print(df, type(df))
print(df.info())

df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', 
                 sep = '\s+')
print(df, type(df))
print(df.info())

df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt',
                 widths=(10, 3, 5), header=None, names=('date', 'name', 'price'), encoding='utf-8')
print(df, type(df))
print(df.info())    

# url = "https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4"
# df = pd.read_html(url, encoding='utf-8')
# print(df)
# print(f"총 {len(df)}개의 테이블이 있습니다.")

# 청크: 대량의 데이터 파일을 읽는 경우, chunk 단위로 읽어 메모리를 사용할 수 있다.
# 장점: 
# - 메모리 사용량을 줄이고, 대량의 데이터를 효율적으로 처리할 수 있다.
# - 스트리밍 방식으로 순차 처리(로그 분석, 실시간 데이터 처리, 머신 러닝 데이터 처리 등)에 유용하다.
# - 분산 처리(batch)
# 단점:
# - 속도가 느리다.

import time
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Malgun Gothic')

n_rows = 10000
# data = {
#     'id': range(1, n_rows + 1),
#     'name': [f'Student_{i}' for i in range(1, n_rows + 1)],
#     'score1': np.random.randint(50, 101, n_rows),
#     'score2': np.random.randint(50, 101, n_rows)
# }

# df = pd.DataFrame(data)
# print(df.head(3))
# print(df.tail(3))
# csv_path = 'students.csv'
# df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 전체 한 번에 처리하는 시간 측정
start_all = time.time()
df_all = pd.read_csv('students.csv', encoding='utf-8-sig')
time_all = time.time() - start_all
print(df_all)

avg_score1 = (df_all['score1'] + df_all['score2']) / 2
print(f"전체 평균 점수: {avg_score1.mean():.2f}")
avg_score2 = (df_all['score2'] + df_all['score2']) / 2
print(f"전체 평균 점수: {avg_score2.mean():.2f}")

# 청크 단위로 처리
chunk_size = 1000
total_score1 = 0
total_score2 = 0
total_count = 0

start_chunk_total = time.time()
for i, chunk in enumerate(pd.read_csv('students.csv', chunksize=chunk_size)):
    start_chunk = time.time()
    # 청크 처리할 때 마다 첫번째 학생 정보만 출력
    if i != 0:
        first_student = chunk.iloc[0]
        print(f"Chunk {i + 1}:")
        print(f"첫번째 학생 id={first_student['id']}")
        print(f"이름={first_student['name']}, ")
        print(f"점수1={first_student['score1']}")
        print(f"점수2={first_student['score2']}")

    total_score1 += chunk['score1'].sum()
    total_score2 += chunk['score2'].sum()
    total_count += len(chunk)
    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f"청크 {i + 1} 처리 시간: {elapsed:.4f}초")

time_chunk_total = time.time() - start_chunk_total
average_score1 = total_score1 / total_count # score1의 전체 평균
average_score2 = total_score2 / total_count # score2의 전체 평균
print(f"전체 청크 처리 시간: {time_chunk_total:.4f}초")
print(f"전체 평균 점수1: {average_score1:.2f}")
print(f"전체 평균 점수2: {average_score2:.2f}")

print('\n처리 결과 요약')
print(f"전체 학생 수: {total_count}")
print(f"Score1 총합: {total_score1}, 평균: {average_score1:.2f}")
print(f"Score2 총합: {total_score2}, 평균: {average_score2:.2f}")

print(f'전체 한 번에 처리 한 경우 소요 시간: {time_all:.4f}초')
print(f"청크 단위로 처리한 경우 소요 시간: {time_chunk_total:.4f}초")

# 시각화
labels = ['전체 한번에 처리', '청크 단위로 처리']
times = [time_all, time_chunk_total]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue', 'yellow'])

for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{time_val:.4f}초', 
             ha='center', va='bottom', fontsize=10)

plt.ylabel('처리시간(초)')

plt.title('전체 vs 청크 단위 처리 시간 비교')

plt.grid(alpha = 0.5)
plt.tight_layout()
plt.show()