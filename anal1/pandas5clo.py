# 파일 입출력에 대하여 공부한다 
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import time
import matplotlib.pyplot as plt 
plt.rc('font', family='malgun gothic')

print("=== Pandas 파일 입출력 학습 ===")

# 웹에 있는 내용은 url을 주세요 로우로 읽어야 합니다! 
print("\n=== 기본 CSV 읽기 ===")
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df)

# 웹에서 CSV 도 가져올 수도 있고 꼭 CSV로만 읽을 수 있는 것도 아닙니다. 
# 기본적으로 CSV 는 가장 첫번째 데이터를 열의 이름으로 쓰고 있다. 
# 열의 이름으로 쓰고 싶지 않을때는 
print("\n=== 헤더 없이 읽기 ===")
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)
# 이렇게 하면 첫번째 데이터를 헤더로 쓰지 않습니다. 
print(df)

# 열 이름을 넣고 싶으면 넣으면 됨 
print("\n=== 열 이름 지정하기 ===")
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a','b'])
# 열을 오른쪽 부터 채워나가고 있음. 제대로 줄 것 같으면 abcd 
print(df)

print("\n=== 행 스킵하기 ===")
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a','b','c','d','m','s','g'], skiprows=1)
# 행을 빼고 싶으면 스킵 로우스 하면 됨. 
print(df)

print("\n=== 텍스트 파일 읽기 ===")
try:
    df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt')
    print(df)
    print(df.info())
except Exception as e:
    print(f"텍스트 파일 읽기 오류: {e}")

# 정규표현식 인식이 너무 중요합니다. 레귤러 익스프레션, 정규표현식은 언어마다 달라집니다. 
# 다양한 정규 표현식 
print("\n=== 정규표현식으로 읽기 ===")
try:
    df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', 
                       sep='\s+')
    # 정규표현식에 플러스 물음표 등이 의미하는 바를 알아야한다. 
    print(df)
    print(df.info())
except Exception as e:
    print(f"정규표현식 읽기 오류: {e}")

# 폭이 일정한 데이터는 fwf 를 사용합니다.
print("\n=== 고정폭 파일 읽기 ===")
try:
    df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt', 
                     widths=(10,3,5), names=('date','name','price'), encoding='utf-8')
    # 자리수로 끊어서 읽어봅니다.
    print(df)
    print(df.info())
except Exception as e:
    print(f"고정폭 파일 읽기 오류: {e}")

# 웹 테이블 읽기
print("\n=== 웹 테이블 읽기 ===")
try:
    url = "https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4" 
    # 인코딩 된 데이터 반드시 시리얼라이징 해줍니다. 인코딩 디코딩 있음 
    df_list = pd.read_html(url)
    print(f"총 {len(df_list)}개의 테이블")
    if df_list:
        print("첫 번째 테이블 미리보기:")
        print(df_list[0].head())
except Exception as e:
    print(f"웹 테이블 읽기 오류: {e}")

# 청크에 대한 이야기를 합니다. 아주 중요한 이야기에요. 덩어리라는 이야기입니다. 
# 대량의 데이터를 쪼갰을 때 청크라고 이야기를 합니다. 
# 1) 대량의 데이터 파일을 읽는 경우 chunk 단위로 분리해 읽기 가능 
# 2) 메모리를 절약할 수 있습니다. 스트리밍 방식으로 순차적으로 처리 할 수 있습니다. (로그 분석, 실시간 데이터 분석)
# 3) 분산처리(batch) 속도는 좀 떨어질 수 있음 

print("\n=== 청크 처리용 데이터 생성 ===")
n_rows = 10000
data = { 
    'id': range(1, n_rows + 1),
    'name': [f'Student_{i}' for i in range(1, n_rows + 1)],
    'score1': np.random.randint(50, 101, size=n_rows),
    'score2': np.random.randint(50, 101, size=n_rows)
}

df = pd.DataFrame(data)
print(df.head(3))
print(df.tail(3))
csv_path = 'students.csv'  # 수정: student.csv -> students.csv
df.to_csv(csv_path, index=False)  # 수정: to_scv -> to_csv
print(f"데이터를 {csv_path}에 저장했습니다.")

# 작성된 csv 파일 사용 : 전체 한방에 처리하기 
print("\n=== 전체 데이터 한 번에 처리 ===")
start_all = time.time()
df_all = pd.read_csv('students.csv')
print(f"전체 데이터 행 수: {len(df_all):,}")

average_all_1 = df_all['score1'].mean() 
average_all_2 = df_all['score2'].mean() 
time_all = time.time() - start_all
print(f"전체 처리 시간: {time_all:.4f}초")
print(f"전체 Score1 평균: {average_all_1:.2f}")
print(f"전체 Score2 평균: {average_all_2:.2f}")

# 이번엔 청크로 처리하겠습니다. 
# chunk로 처리하기 
print("\n=== 청크로 처리하기 ===")
chunk_size = 1000
total_score1 = 0
total_score2 = 0  # 추가
total_count = 0

start_chunk_total = time.time()

# 수정된 for 루프
for i, chunk in enumerate(pd.read_csv('students.csv', chunksize=chunk_size)):  # 문법 수정
    start_chunk = time.time()
    
    # 청크 처리 할 때 마다 첫번째 학생 정보만 출력한다.
    first_student = chunk.iloc[0]
    print(f"chunk {i+1} 첫번째 학생: id = {first_student['id']}, "
          f"이름 = {first_student['name']}, "
          f"score1 = {first_student['score1']}, "
          f"score2 = {first_student['score2']}")
    
    # 누적 계산
    total_score1 += chunk['score1'].sum()
    total_score2 += chunk['score2'].sum()  # 수정: score2 제대로 계산
    total_count += len(chunk)
    
    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f'처리시간: {elapsed:.4f}초')
    print()

time_chunk_total = time.time() - start_chunk_total  # 문법 수정

# 청크 처리 결과 계산
chunk_average1 = total_score1 / total_count
chunk_average2 = total_score2 / total_count

print('=== 처리결과 요약 ===')
print(f'전체 학생 수: {total_count:,}명')  # 문법 수정
print(f'score1 총합: {total_score1:,}, 평균: {chunk_average1:.2f}')  # 문법 수정
print(f'score2 총합: {total_score2:,}, 평균: {chunk_average2:.2f}')  # 문법 수정

print(f'\n전체 한번에 처리한 경우 소요시간: {time_all:.4f}초')  # 문법 수정
print(f'청크로 처리한 경우 전체 소요시간: {time_chunk_total:.4f}초')  # 문법 수정

# 결과 비교
print(f'\n=== 성능 및 정확도 비교 ===')
print(f'시간 차이: {abs(time_all - time_chunk_total):.4f}초')
print(f'Score1 평균 차이: {abs(average_all_1 - chunk_average1):.6f}')
print(f'Score2 평균 차이: {abs(average_all_2 - chunk_average2):.6f}')

if time_chunk_total > time_all:
    print(f'청크 처리가 {time_chunk_total - time_all:.4f}초 더 오래 걸렸습니다.')
else:
    print(f'청크 처리가 {time_all - time_chunk_total:.4f}초 더 빨랐습니다.')

# 시각화 
print("\n=== 성능 비교 시각화 ===")
labels = ['전체 한번에 처리', '청크로 처리']
times = [time_all, time_chunk_total]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, times, color=['skyblue', 'yellow'])

# 막대 위에 시간 표시
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, 
             f'{time_val:.4f}초', ha='center', va='bottom', fontsize=12)

plt.ylabel('처리시간(초)')
plt.title('전체 한번에 처리 vs 청크로 처리 - 성능 비교')
plt.grid(alpha=0.3)
plt.tight_layout()

# 그래프 저장 (선택사항)
plt.savefig('chunk_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 추가 통계 정보 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 평균 점수 비교
methods = ['전체 처리', '청크 처리']
score1_avgs = [average_all_1, chunk_average1]
score2_avgs = [average_all_2, chunk_average2]

x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, score1_avgs, width, label='Score1', color='lightcoral')
ax1.bar(x + width/2, score2_avgs, width, label='Score2', color='lightgreen')
ax1.set_xlabel('처리 방법')
ax1.set_ylabel('평균 점수')
ax1.set_title('처리 방법별 평균 점수 비교')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend()
ax1.grid(alpha=0.3)

# 정확도 차이 시각화
accuracy_diff = [abs(average_all_1 - chunk_average1), abs(average_all_2 - chunk_average2)]
score_types = ['Score1 차이', 'Score2 차이']

ax2.bar(score_types, accuracy_diff, color=['orange', 'purple'])
ax2.set_ylabel('평균값 차이')
ax2.set_title('전체 처리 vs 청크 처리 정확도 차이')
ax2.grid(alpha=0.3)

# 차이값 표시
for i, diff in enumerate(accuracy_diff):
    ax2.text(i, diff + 0.001, f'{diff:.6f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

print("\n✅ 모든 파일 입출력 및 청크 처리 학습이 완료되었습니다!")
print("📊 성능 비교 그래프가 생성되었습니다!")
print(f"📁 그래프가 'chunk_performance_comparison.png'로 저장되었습니다!")