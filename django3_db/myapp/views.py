from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd

def indexFunc(request):
    return render(request, 'index.html')

def dbshowFunc(request):
    dept = (request.GET.get('dept') or "").strip()

    sql = """
    SELECT
        j.jikwonno   AS 직원번호,
        j.jikwonname AS 직원명,
        b.busername  AS 부서명,
        b.busertel   AS 부서전화,
        j.jikwonjik  AS 직급,
        j.jikwonpay  AS 연봉
    FROM jikwon j
    INNER JOIN buser b ON j.busernum = b.buserno
    """
    params = []
    if dept:
        sql += " WHERE b.busername LIKE %s"
        params.append(f"%{dept}%")

    sql += " ORDER BY j.jikwonno"

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description]

    if rows:
        df = pd.DataFrame(rows, columns=cols)
        # 집계 안전을 위해 숫자 변환
        df['연봉'] = pd.to_numeric(df['연봉'], errors='coerce')

        # 조인 결과 표
        join_html = df[['직원번호', '직원명', '부서명', '부서전화', '직급', '연봉']].to_html(index=False)

        # 직급별 연봉 통계
        stats_df = (
            df.groupby('직급')['연봉']
              .agg(
                  평균='mean',
                  표준편차=lambda x: x.std(ddof=0),
                  인원수='count',
              )
              .round(2)
              .reset_index()
              .sort_values(by='평균', ascending=False)
        )
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        stats_html = stats_df.to_html(index=False)
    else:
        join_html = "조회된 데이터가 없습니다."
        stats_html = "통계 대상 자료가 없습니다."

    ctx = {
        'dept': escape(dept),
        'join_html': join_html,
        'stats_html': stats_html,
    }
    return render(request, 'dbshow.html', ctx)

