#file i/o 파일 입출력에 대하여 공부한다 

import pandas as pd
import numpy as np
from pandas import Series, DataFrame

#df = pd.read_csv('anal1/ex1.csv')
#df = pd.read_csv('anal1/ex1.csv', sep=',') #테이블로 읽을수도 있지만 세퍼레이터를 안주면 통으로 읽는다 테이블로 읽을때는 셉을 필수로 줘야 한다.
#print(df,type(df)) #외부 파일을 읽어옴 
#print(df.info())


#웹에 있는 내용은 url을 주세요 로우로 읽어야 합니다! 
print()
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df)

#웹에서 CSV 도 가져올 수도 있고 꼭 CSV로만 읽을 수 있는 것도 아닙니다. 
#본적으로 CSV 는 가장 첫번째 데이터를 열의 이름으로 쓰고 있다. 
#열의 이름으로 쓰고 싶지 않을떄는 
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)
#이렇게 하면 첫번째 데이터를 헤더로 쓰지 않습니다. 
#열 이름을 넣고 싶으면 넣으면 됨 
print(df)

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None, names=['a','b'])
#열을 오른쪽 부터 채워나가고 있음. 제대로 줄 것 같으면 abcd 
print(df)

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None,
                 names=['a','b','c','d','m','s','g'], skiprows=1) #행을 빼고 싶으면 스킵 로우스 하면 됨. 
#열을 오른쪽 부터 채워나가고 있음. 제대로 줄 것 같으면 abcd 
print(df)

print( )
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt')
print(df)
print(df.info())
#제대로 읽어주려면 세퍼레이터를 줘도 되고 

#정규표 인식이 너무 중요합니다. 레귤러 익스 프레시, 정규표현식은 언어마다 달라집니다. 정규표현식 디테일한 내용이 있으면 좋습니다. 
#다양한 정규 표현식 
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', sep='\s+')
#정규표현식에 플러스 물음표 등이 의미하는 바를 알아야한다. 
#이런식으로 정규 표현식을 쓸 수 있다. 
print(df)
print(df.info()) 

#이번에는 깃허브에 가서 
#폭이 일정한 데이터는 fwf 를 사용합니다.
df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt', widths=(10,3,5), names=('date','name','price'), encoding='utf')
#자리수로 끊어서 읽어봅니다.
#지금은 천천히 보여주고 있지만 대충 알아봅시다.
print(df)
print(df.info()) 
#이런식으로 데이터를 읽을 수 있습니다. 

"""
print()
url = "https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4" 
#인코딩 된 데이터 반드시 시리얼라이징 해줍니다. 인코딩 디코딩 있음 
#마리아 디비는 데이터 프레임으로 잡아오고, 웹에 있는 데이터도 잡아와야 해요 
df = pd.read_html(url)
print(df)
print(f"총{len(df)}개의 자료")
"""
#청크에 대한 이야기를 합니다. 아주 중요한 이야기에요. 덩어리라는 이야기입니다. 대량의 데이터를 쪼갰을 떄 청크라고 이야기를 합니다. 
# 1) 대량의 데이터 파일을 읽는 경우 chunk 단위로 분리해 읽기 가능 
# 2) 메모리를 절약할 수 있습니다. 스트리밍 방식으로 순차적으로 처리 할 수 있습니다. (로그 분석, 실시간 데이터 데이터 분석)
# 3) 분산처리(bach) 속도는 좀 떨어질 수 있음 

import time
import matplotlib.pyplot as plt 
plt.rc('font',family= 'malgun gothic')


""""
n_rows = 10000
data = { 
        'id': range(1, n_rows + 1),
        'name':[f'Student_{i}' for i in range(1, n_rows + 1)],
        'score1' : np.random.randint(50, 101, size=n_rows),
        'score2' : np.random.randint(50, 101, size=n_rows)
        }

df = pd.DataFrame(data)
print(df.head(3))
print(df.tail(3))
csv_path = 'student.csv'
df.to_scv(csv_path, index=False)



#작성된 csv 파일 사용 : 전체 한방에 처리하기 
start_all = time.time()
df_all = pd.read_csv('students.csv')
print(df_all)

average_all_1 = df_all['score1'].mean() 
average_all_2 = df_all['score2'].mean() 
time_all = time.time() -start_all
print(time_all)

#이번엔 청크로 처리하겠습니다. 
#chunk로 처리하기 
chunk_size = 1000
total_score1 = 0
total_count = 0
start_chunk_total = time.time()
for i chunk in enumerate(pd.read_csv('students.csv',chunk_size))
    start_chunk = time.time()
    #청크 처리 할 떄 마다 첫번째 학생 정보만 출력한다.
    #첫번째 학생 천번째 학생 이렇게 
    first_student = chunk.iloc[0]
    print(f"chunk{i+1} 첫번째 학생 : id {first_student['id'], 이름 = {fitst_studert['name']}
                                    f "score1= first_student['score1]"}, score2= {first_student ['score2']}")
    total_sore1 += chunk['score1'].sum()
    total_sore2 += chunk['score1'].sum()
    end_chunk =time.time()
    elapsed = end_chunk - start_chunk
    print ('처리시간 {elapsed}초')

    
time_chunk_total = time.time - start_chunk_total #전체 시간을 읽을 수 있다 

print('\n 처리결과 요약')
print(f '전체 학색 수' : {total_count)})
print(f'scort1 총합 : {total score}', 평군 총힙 :)
print(f'scort2 총합 : {total score}', 평군 총힙 :)

print ('전체 한번에 처리한 경우 소요시간 : time_all.4f')
print ('청크로 처리한 경우 전체 소요시간 처리한 경우 소요시간 : time_chuk_total.4f')


# 시각화 
labels = ['전체 한번에 처리', '청크로 처리']
times = [time_all, time_chunk_total]
plt.figure(figsize=(6,4))
bars = plt.bar(labels, time, color = 'skyble', 'yellow')

for bar, time_val in zip(bar, times):
    plt.text(bar.get_x() + var.get_whith() /2 , bar.get_height(), f'(time_val:4f)초'
        ha = 'center', v= 'bottom', frontsze=0)

plt.ylable('처리시간(초)')
plt.title('전체 한번에 처리 vs청크로 처리')
plt.gride(alpha =0.5)
plt.tight_layout()
plt.show()

"""