import MySQLdb


conn = MySQLdb.connect(
    host='127.0.0.1',
    user='root',
    password='12345',
    database='mydb',
    port=3306,
    charset='utf8'
)

cursor = conn.cursor()

try:
    cursor = conn.cursor()
    no = int(input('사원번호 입력:')) # 사원번호 입력
    name = input('이름 입력:') # 이름 입력

    # DB에 직접 입력받은 no, name로 검색
    sql_login = """
        select jikwonno, buser.busername, jikwonname, jikwonjik, buser.busertel, jikwongen
        from jikwon inner join buser
        on jikwon.busernum=buser.buserno
        where jikwonname = %s and jikwonno = %s 
    """
    cursor.execute(sql_login, (name,no)) # sql_login 실행
    result = cursor.fetchone() # 일치한 행 하나 가져오기
    
    if result:
        print('로그인 성공')
        
    else:
        print('로그인 실패')

except Exception as e:
    print(f'문제 발생 :{e}')

except MySQLdb.OperationalError as e:
    print(f'문제(2) 발생: {e}')

finally:
    cursor.close()
    conn.close()