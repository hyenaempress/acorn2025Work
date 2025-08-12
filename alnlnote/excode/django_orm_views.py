# myapp/views.py - ORM 활용 버전

from django.shortcuts import render
from django.db.models import Count, Avg, Sum, Q, F, Max, Min
from django.db import connection
from django.utils.html import escape
from .models import Jikwon, Buser, Gogek
import pandas as pd

def ormAnalysisFunc(request):
    """Django ORM을 활용한 직원 정보 분석"""
    
    # 사용자로부터 부서명 받기
    dept = (request.GET.get('dept') or "").strip()
    safe_dept = escape(dept)

    # =================================================================
    # 1. 기본 ORM 조회 - Raw SQL 대신 ORM 사용
    # =================================================================
    
    # Raw SQL: SELECT j.*, b.busername FROM jikwon j INNER JOIN buser b ON j.busernum = b.buserno
    # ORM 버전: select_related()를 사용한 JOIN
    jikwons_query = Jikwon.objects.select_related('busernum')
    
    # 부서 필터링 (Raw SQL의 WHERE 절과 동일)
    if dept:
        # Raw SQL: WHERE b.busername LIKE '%dept%'
        # ORM 버전: filter() 메서드 사용
        jikwons_query = jikwons_query.filter(
            busernum__busername__icontains=dept  # LIKE '%dept%'와 동일
        )
    
    # 정렬 (Raw SQL의 ORDER BY와 동일)
    jikwons = jikwons_query.order_by('jikwonno')
    
    # =================================================================
    # 2. ORM 집계 함수 활용 - 부서별 통계
    # =================================================================
    
    # Raw SQL: SELECT busername, COUNT(*), AVG(jikwonpay) FROM ... GROUP BY busername
    # ORM 버전: annotate()와 aggregate() 사용
    dept_stats = (
        Buser.objects
        .annotate(
            직원수=Count('jikwon'),  # COUNT(*) 
            평균연봉=Avg('jikwon__jikwonpay'),  # AVG(jikwonpay)
            총연봉=Sum('jikwon__jikwonpay'),  # SUM(jikwonpay)
            최고연봉=Max('jikwon__jikwonpay'),  # MAX(jikwonpay)
            최저연봉=Min('jikwon__jikwonpay')   # MIN(jikwonpay)
        )
        .filter(직원수__gt=0)  # 직원이 있는 부서만
        .order_by('-평균연봉')  # 평균연봉 내림차순
    )
    
    # =================================================================
    # 3. 복잡한 조건 쿼리 - Q 객체 활용
    # =================================================================
    
    # Raw SQL: WHERE (jikwonpay > 4000 AND jikwonjik = '부장') OR jikwongen = '남'
    # ORM 버전: Q 객체로 복잡한 조건 구성
    high_performers = Jikwon.objects.filter(
        Q(jikwonpay__gt=4000, jikwonjik='부장') |  # AND 조건
        Q(jikwongen='남')  # OR 조건
    ).select_related('busernum')
    
    # =================================================================
    # 4. F 객체를 활용한 필드 간 연산
    # =================================================================
    
    # 근속년수 계산 (현재 날짜 - 입사일)
    from django.utils import timezone
    from datetime import date
    
    # F 객체로 데이터베이스 레벨에서 계산
    jikwons_with_years = Jikwon.objects.annotate(
        근속년수=timezone.now().year - F('jikwonibsail__year')
    ).select_related('busernum')
    
    # =================================================================
    # 5. 서브쿼리와 EXISTS 활용
    # =================================================================
    
    # 고객을 담당하는 직원만 조회
    # Raw SQL: SELECT * FROM jikwon WHERE EXISTS (SELECT 1 FROM gogek WHERE gogekdamsano = jikwonno)
    # ORM 버전: Exists와 Subquery 사용
    from django.db.models import Exists, OuterRef
    
    has_customers = Gogek.objects.filter(gogekdamsano=OuterRef('jikwonno'))
    jikwons_with_customers = Jikwon.objects.filter(
        Exists(has_customers)
    ).select_related('busernum')
    
    # =================================================================
    # 6. 직급별 분석 - values()와 annotate() 조합
    # =================================================================
    
    # Raw SQL: SELECT jikwonjik, COUNT(*), AVG(jikwonpay) FROM jikwon GROUP BY jikwonjik
    # ORM 버전:
    position_analysis = (
        Jikwon.objects
        .values('jikwonjik')  # GROUP BY jikwonjik
        .annotate(
            인원수=Count('jikwonno'),
            평균연봉=Avg('jikwonpay'),
            연봉합계=Sum('jikwonpay')
        )
        .order_by('-평균연봉')
    )
    
    # =================================================================
    # 7. 데이터 처리 및 템플릿용 변환
    # =================================================================
    
    # 1. 직원 목록을 DataFrame으로 변환
    jikwon_data = []
    for j in jikwons:
        jikwon_data.append({
            '직원번호': j.jikwonno,
            '직원명': j.jikwonname,
            '부서명': j.busernum.busername if j.busernum else '미배정',
            '부서전화': j.busernum.busertel if j.busernum else '',
            '직급': j.jikwonjik or '',
            '연봉': j.jikwonpay or 0
        })
    
    if jikwon_data:
        df_jikwon = pd.DataFrame(jikwon_data)
        join_html = df_jikwon.to_html(
            index=False,
            classes='table table-striped table-hover'
        )
    else:
        join_html = "<p class='text-center text-muted'>조건에 맞는 직원이 없습니다.</p>"
    
    # 2. 부서별 통계를 DataFrame으로 변환
    dept_data = []
    for dept_stat in dept_stats:
        dept_data.append({
            '부서명': dept_stat.busername,
            '직원수': dept_stat.직원수,
            '평균연봉': round(dept_stat.평균연봉 or 0),
            '총연봉': dept_stat.총연봉 or 0,
            '최고연봉': dept_stat.최고연봉 or 0,
            '최저연봉': dept_stat.최저연봉 or 0
        })
    
    if dept_data:
        df_dept = pd.DataFrame(dept_data)
        dept_html = df_dept.to_html(
            index=False,
            classes='table table-info table-striped'
        )
    else:
        dept_html = "<p class='text-center text-muted'>부서 데이터가 없습니다.</p>"
    
    # 3. 직급별 분석을 DataFrame으로 변환
    position_data = []
    for pos in position_analysis:
        position_data.append({
            '직급': pos['jikwonjik'] or '미지정',
            '인원수': pos['인원수'],
            '평균연봉': round(pos['평균연봉'] or 0),
            '연봉합계': pos['연봉합계'] or 0
        })
    
    if position_data:
        df_position = pd.DataFrame(position_data)
        position_html = df_position.to_html(
            index=False,
            classes='table table-success table-striped'
        )
    else:
        position_html = "<p class='text-center text-muted'>직급 데이터가 없습니다.</p>"
    
    # =================================================================
    # 8. 고급 ORM 기법 - 조건부 집계
    # =================================================================
    
    # Case When 사용 - 성별에 따른 조건부 집계
    from django.db.models import Case, When, IntegerField
    
    gender_stats = Jikwon.objects.aggregate(
        남성직원수=Count(
            Case(When(jikwongen='남', then=1), output_field=IntegerField())
        ),
        여성직원수=Count(
            Case(When(jikwongen='여', then=1), output_field=IntegerField())
        ),
        남성평균연봉=Avg(
            Case(When(jikwongen='남', then='jikwonpay'), output_field=IntegerField())
        ),
        여성평균연봉=Avg(
            Case(When(jikwongen='여', then='jikwonpay'), output_field=IntegerField())
        )
    )
    
    # 템플릿으로 전달할 컨텍스트
    context = {
        'join_html': join_html,
        'dept_html': dept_html,
        'position_html': position_html,
        'dept': safe_dept,
        'total_count': len(jikwon_data),
        'dept_count': len(dept_data),
        'gender_stats': gender_stats,
        'high_performers_count': high_performers.count(),
        'customers_staff_count': jikwons_with_customers.count(),
    }
    
    return render(request, 'orm_analysis.html', context)

# =================================================================
# Django ORM 기본 원칙과 패턴 가이드
# =================================================================

"""
🔥 Django ORM 핵심 원칙 및 베스트 프랙티스

1. LAZY EVALUATION (지연 평가)
   - QuerySet은 실제로 데이터가 필요할 때까지 DB에 접근하지 않음
   - 체이닝 가능: .filter().order_by().select_related()

2. 주요 QuerySet 메서드:
   
   📌 기본 조회
   - .all()          : 전체 조회
   - .get()          : 단일 객체 조회 (없으면 DoesNotExist 예외)
   - .filter()       : 조건 필터링 (WHERE)
   - .exclude()      : 조건 제외
   - .first(), .last() : 첫 번째/마지막 객체
   
   📌 정렬
   - .order_by()     : 정렬 (ORDER BY)
   - .order_by('-field') : 내림차순
   
   📌 조인 최적화
   - .select_related() : INNER JOIN (1:1, N:1 관계)
   - .prefetch_related() : 별도 쿼리로 관련 객체 미리 로드 (1:N, M:N)
   
   📌 집계
   - .aggregate()    : 전체 집계 (딕셔너리 반환)
   - .annotate()     : 각 객체별 집계 (QuerySet 반환)
   - .values()       : 특정 필드만 조회 (딕셔너리 형태)
   - .values_list()  : 튜플 형태로 조회
   
   📌 조건 구성
   - Q()             : 복잡한 조건 (AND, OR, NOT)
   - F()             : 필드 간 비교, 연산
   - Case(), When()  : 조건부 표현식

3. 필드 룩업 (Field Lookups):
   - __exact         : 정확히 일치
   - __iexact        : 대소문자 무시 일치
   - __contains      : 포함 (LIKE '%value%')
   - __icontains     : 대소문자 무시 포함
   - __startswith    : 시작 (LIKE 'value%')
   - __endswith      : 끝 (LIKE '%value')
   - __gt, __gte     : 초과, 이상
   - __lt, __lte     : 미만, 이하
   - __in            : 리스트 내 포함 (IN)
   - __isnull        : NULL 체크
   - __year, __month : 날짜 필드의 연도, 월

4. 성능 최적화 팁:
   ✅ select_related() 사용으로 N+1 문제 해결
   ✅ only(), defer() 로 필요한 필드만 조회
   ✅ exists() 로 존재 여부만 확인
   ✅ count() 대신 len()을 적절히 활용
   ✅ bulk_create(), bulk_update() 로 대량 처리

5. 안티패턴 (피해야 할 것들):
   ❌ 반복문에서 개별 쿼리 실행 (N+1 문제)
   ❌ .all()로 전체 조회 후 파이썬에서 필터링
   ❌ 불필요한 ORDER BY
   ❌ 관련 객체 접근 시 select_related() 미사용

6. Raw SQL과의 비교:
   Raw SQL: 직접적, 복잡한 쿼리 가능, 데이터베이스 종속적
   ORM: 안전, 가독성, 데이터베이스 독립적, 객체지향적

강사님 조언: "편한 걸로 하시면 됩니다" 
→ 상황에 맞게 ORM과 Raw SQL을 적절히 혼용하는 것이 베스트!
"""

def ormBasicExamples(request):
    """ORM 기본 사용법 예제 모음"""
    
    # 1. 기본 CRUD
    # CREATE
    # new_jikwon = Jikwon.objects.create(
    #     jikwonname='홍길동',
    #     jikwonjik='사원',
    #     jikwonpay=3000
    # )
    
    # READ
    all_jikwons = Jikwon.objects.all()
    specific_jikwon = Jikwon.objects.get(jikwonno=1)
    filtered_jikwons = Jikwon.objects.filter(jikwonjik='부장')
    
    # UPDATE
    # Jikwon.objects.filter(jikwonno=1).update(jikwonpay=5000)
    
    # DELETE
    # Jikwon.objects.filter(jikwonno=1).delete()
    
    # 2. 복잡한 조건
    complex_query = Jikwon.objects.filter(
        Q(jikwonpay__gte=4000) & Q(jikwonjik='부장') |
        Q(jikwongen='여')
    )
    
    # 3. 집계 쿼리
    stats = Jikwon.objects.aggregate(
        avg_pay=Avg('jikwonpay'),
        max_pay=Max('jikwonpay'),
        count=Count('jikwonno')
    )
    
    return render(request, 'orm_examples.html', {
        'all_jikwons': all_jikwons,
        'filtered_jikwons': filtered_jikwons,
        'complex_query': complex_query,
        'stats': stats
    })