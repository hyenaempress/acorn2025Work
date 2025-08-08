import sqlite3

# 1. DB 연결 및 테이블 생성
conn = sqlite3.connect(':memory:')  # 실험용 메모리 DB
conn.execute("""
    create table if not exists test(
        product varchar(10),
        maker varchar(10),
        weight real,
        price integer
    );
""")
conn.commit()

# 2. 데이터 삽입
data1 = ('mouse', 'samsung', 12.5, 5000)
stmt = "insert into test values(?, ?, ?, ?)"
conn.execute(stmt, data1)

# 3. 두 번째 데이터도 삽입 (오타 수정)
data2 = ('mouse2', 'samsung', 15, 5000)
conn.execute(stmt, data2)

conn.commit()

# 4. 데이터 조회
cursor = conn.cursor()
cursor.execute("select * from test")
rows = cursor.fetchall()

# 5. 안전하게 출력
for i, row in enumerate(rows):
    print(f"{i+1}번째 행:", row)

