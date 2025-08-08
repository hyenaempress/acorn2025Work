#Local Database 연동 후 자료를 읽어 Datafame에 저장한다.

import sqlite3


#컬럼은 네개짜리 램에만 실험용으로존재하게 하겠습니다
#실험용 입니다 
#test db  라고 하면 파일로 만들어집니다
sql = "create table if not exists test(product varchar(10), maker varchar(10), weight real, price integer);"

conn = sqlite3.connect(':memory:')  # 메모리 DB
conn.execute(sql)
conn.commit()

#한개씩 추가 
#data1 = ('mouse', 'samsung', 12.5, 5000)    (4)  (4,) #가운데 있는 (4)는 튜플이 아니다. 데이터는 꼭 콤마를 찍어줘야한다. 

stmt = "insert into test values(?, ?, ?, ?)" #이런식으로 물음표를 써서 맵핑을 해야한다. 안그럼 시큐어 코딩에 위배됩니다. 가이드라인에 걸립니다. 
#물음표 연산자 활용 맵핑이 필요 합니다.
data1 = ('mouse', 'samsung', 12.5, 5000)# 저 위에것을 수행에 참여시키면 된다 
data2 = ('mouse2', 'samsung', 15, 5000)
conn.execute(stmt, data1) #데이터를집어넣었습니다. 
conn.execute(stmt, data2)

# 복수개 추가 

datas = [('mouse3', 'lg', 22.5, 5000), ('mouse4', '1g', 18.5, 5000)]
conn.executemany(stmt, datas)

#트라이 익셉트 문장을 꼭 써줘야 하는데 지금은 예자 파일이라서 트라이익셉트는 빼고 있습니다. 

cuser = conn.cursor()
cuser.execute("select * from test")
rows = cuser.fetchall()
#print(rows[0], '' , rows[1], rows[0][0])



for a in rows:
    print(a)
    
    
# 파이썬 시간에 이정도 까지 했다면 
# 판다스는 이렇게 합니다.
    
import pandas as pd
df = pd.DataFrame(rows, columns=['product', 'maker', 'weight', 'price'])
print(df) 
#print(df.to_html()) #이런식으로  HTML로 보낼 수 있음  장고 이런데서 

print()
df2 = pd.read_sql("select * from test", conn) # 맨 마지막에 커넥션 객체 넣으면 리드 에스큐엘 넣으면 바로 데이터 프레임으로 갑니다.
print(df2)

#이런 방법으로도 운영이가능합니다.
#이건 매번 실행할때마다 테이블 메모리에 저장됩니다. 

#DB 의 내용을 읽어서 데이터 프레임에 넣었다면, 꺼꾸로 데이터 프레임에 있는 내용을 DB 에 저장할 수 있어야 한다.

#똑같은 내용하면 재미없으니 새로운거 

pdata = {
    'product': ['연필', '볼펜', '지우개'],
    'maker': ['모나미', '모나미', '모나미'],
    'weight': [10, 20, 30],
    'price': [1000, 2000, 3000]
}

frame = pd.DataFrame(pdata)

#꺼꾸로 DB에 넣고 싶으면 어떻게 해야 할까요? 

frame.to_sql('test', conn, if_exists='append', index=False) # 이런식으로 넣을 수 있습니다. 




cuser.close()
conn.close()



