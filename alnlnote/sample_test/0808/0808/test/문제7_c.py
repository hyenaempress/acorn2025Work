'''
 c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
      조건 :  try ~ except MySQLdb.OperationalError as e:      사용
     사번  직원명  부서명   직급  부서전화  성별
     ...
     인원수 : * 명
'''
import MySQLdb
import pandas as pd

conn = MySQLdb.connect(
    host='127.0.0.1',
    user='root',
    password='1234',
    database='mydb',
    port=3306,
    charset='utf8'
)

try:    
    cursor = conn.cursor()
    no = int(input('사원번호 입력:')) # 사원번호 입력
    name = input('이름 입력:') # 이름 입력

    # DB에 입력받은 no, name로 검색
    sql_login = """
        select jikwonno, jikwonname
        from jikwon
        where jikwonname = %s and jikwonno = %s 
    """
    cursor.execute(sql_login, (name,no)) # 명령어 실행/ 입력 받은 name,no 맵핑
    result = cursor.fetchone() # 일치한 행 하나 가져오기
    
    if result: # 일치한 값이 있을 때 실행
        print('로그인 성공')
        
        # 전체 조회할 수 있는 sql 명령어
        sql_all="""
            select jikwonno, jikwonname, buser.busername, jikwonjik, buser.busertel, jikwongen
            from jikwon inner join buser
            on jikwon.busernum=buser.buserno
        """
        cursor.execute(sql_all) # 명령어 실행
        # DataFrame에 담기
        df = pd.DataFrame(cursor.fetchall(), columns=[
            '사번', '직원명','부서명','직급', '부서전화','성별'
        ])
        print(df.head(3))
        print(f'인원수 :{df["사번"].count()}명') # 직원 전체 인원수 출력
        
    else:
        print('로그인 실패')

except MySQLdb.OperationalError as e: # DB접속, 인증, 네트워크 문제
    print(f'문제(2) 발생: {e}')

except Exception as e:
    print(f'문제 발생 :{e}')

finally:
    cursor.close()
    conn.close()