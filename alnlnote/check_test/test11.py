import MySQLdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import sys

def main():
    CONFIG = {
        "host": "127.0.0.1",
        "user": "root",
        "passwd": "123",
        "db": "test",
        "port": 3306,
        "charset": "utf8",
    }

    sql = """
     SELECT jikwonno, jikwonname, jikwonpay
    FROM jikwon
    WHERE jikwonno NOT IN (
        SELECT DISTINCT gogek_damsano 
        FROM gogek 
        WHERE gogek_damsano IS NOT NULL
    )
    """

    try:
        conn = MySQLdb.connect(**CONFIG) ## 디비 연결
        cursor = conn.cursor()
     
        df = pd.read_sql(sql, conn)
        
        count = len(df)
        salary_mean = df['jikwonpay'].mean()
        salary_median = df['jikwonpay'].median()
        salary_std = df['jikwonpay'].std()
        
        print(f"직원 수: {count}")
        print(f"평균 급여: {salary_mean}")
        print(f"중앙값 급여: {salary_median}")
        print(f"표준편차 급여: {salary_std}")
        
    # 히스토그램 출력 
    
        plt.figure(figsize=(10, 6))
        plt.hist(df['jikwonpay'], bins=10, color='skyblue', edgecolor='black')
        plt.title('급여 분포')
        plt.xlabel('급여')
        plt.ylabel('빈도')
        plt.grid(True)
        plt.show()
    
        
    except MySQLdb.Error as e:
        print(f"디비오류: {e}")
    except Exception as e:
        print(f"처리 오류: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
   
if __name__ == "__main__":
    main()