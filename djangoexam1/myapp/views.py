from django.shortcuts import render
from django.db import connection
from .models import Jikwon# 또는: from myapp.models import Jikwon
import pandas as pd
from django.utils.html import escape
from datetime import date



def indexFunc(request):
    """메인 페이지"""
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


# myapp/views.py
from datetime import date
import pandas as pd
from django.shortcuts import render
from django.utils.html import escape
from .models import Jikwon

def ormAnalysisFunc(request):
    """ORM을 사용한 직원 정보 조회"""
    dept = (request.GET.get('dept') or "").strip()

    # 0) ORM 쿼리: 조인 + 필터 + 정렬
    qs = (
        Jikwon.objects
        .select_related('busernum')
    )
    if dept:
        qs = qs.filter(busernum__busername__icontains=dept)

    # 정렬 요구: "부서번호, 직원명 오름차순" (없으면 사번으로)
    qs = qs.order_by('busernum__buserno', 'jikwonname')

    # 1) values()로 DataFrame 만들기 좋은 형태로 뽑기
    rows = qs.values(
        'jikwonno',               # 직원번호
        'jikwonname',             # 직원명
        'busernum__busername',    # 부서명
        'busernum__busertel',     # 부서전화
        'jikwonjik',              # 직급
        'jikwonpay',              # 연봉
        'jikwongen',              # 성별
        'jikwonibsail',           # 입사일(근무년수 계산용)
    )

    df = pd.DataFrame(list(rows))
    if df.empty:
        # 결과가 없으면 빈 표들로 처리
        empty_html = "<p>검색 결과가 없습니다.</p>"
        ctx = {
            'dept': escape(dept),
            'join_html': empty_html,
            'dept_pos_pay_html': empty_html,
            'dept_pay_html': empty_html,
            'gender_pos_freq_html': empty_html,
        }
        return render(request, 'myapp/orm_result.html', ctx)

    # 컬럼 한글화 + 근무년수 계산
    df = df.rename(columns={
        'jikwonno': '직원번호',
        'jikwonname': '직원명',
        'busernum__busername': '부서명',
        'busernum__busertel': '부서전화',
        'jikwonjik': '직급',
        'jikwonpay': '연봉',
        'jikwongen': '성별',
        'jikwonibsail': '입사일',
    })

    # 근무년수(만 연수): 입사일 결측 방어
    today = date.today()
    def years_of_service(d):
        try:
            if pd.isna(d):
                return None
            # d가 datetime.date/Datetime/str 다양한 경우가 있으니 변환
            dt = pd.to_datetime(d).date()
            years = today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
            return max(years, 0)
        except Exception:
            return None

    df['근무년수'] = df['입사일'].apply(years_of_service)

    # NaN/None 정리 (표시용)
    df_display = df[['직원번호', '직원명', '부서명', '부서전화', '직급', '연봉', '근무년수']].fillna('')

    # (1) 직원 기본 조인 결과 표
    join_html = df_display.to_html(index=False)

    # (2) 부서명, 직급 기준 연봉 합/평균
    dept_pos = (
        df.groupby(['부서명', '직급'], dropna=False)['연봉']
          .agg(['sum', 'mean'])
          .reset_index()
    )
    dept_pos_pay_html = dept_pos.fillna(0).to_html(index=False)

    # (3) 부서명별 연봉 합/평균
    dept_pay = (
        df.groupby(['부서명'], dropna=False)['연봉']
          .agg(['sum', 'mean'])
          .reset_index()
    )
    dept_pay_html = dept_pay.fillna(0).to_html(index=False)

    # (4) 성별, 직급별 빈도표 (count)
    gender_pos = (
        df.groupby(['성별', '직급'], dropna=False)['직원번호']
          .agg(['count'])
          .reset_index()
          .rename(columns={'count': '인원수'})
    )
    gender_pos_freq_html = gender_pos.fillna(0).to_html(index=False)

    ctx = {
        'dept': escape(dept),
        'join_html': join_html,                       # (1)
        'dept_pos_pay_html': dept_pos_pay_html,       # (2)
        'dept_pay_html': dept_pay_html,               # (3)
        'gender_pos_freq_html': gender_pos_freq_html, # (4)
    }
    return render(request, 'orm_analysis.html', ctx)