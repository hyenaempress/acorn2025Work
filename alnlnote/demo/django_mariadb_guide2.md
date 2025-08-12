# 20.03 Django MariaDB 연결 및 데이터베이스 활용 가이드

## 목차
1. [프로젝트 생성 및 기본 설정](#프로젝트-생성-및-기본-설정)
2. [MariaDB 데이터베이스 설정](#mariadb-데이터베이스-설정)
3. [모델 생성 및 마이그레이션](#모델-생성-및-마이그레이션)
4. [뷰 함수 구현](#뷰-함수-구현)
5. [템플릿 구성](#템플릿-구성)
6. [URL 라우팅](#url-라우팅)
7. [완성된 애플리케이션](#완성된-애플리케이션)
8. [트러블슈팅](#트러블슈팅)

---

## 프로젝트 생성 및 기본 설정

### 1. 프로젝트 구조 생성

```bash
# 작업 디렉토리 생성
PS D:\work\acorn2025Work> mkdir django3_db
PS D:\work\acorn2025Work> cd django3_db

# Django 프로젝트 생성
PS D:\work\acorn2025Work\django3_db> django-admin startproject mainapp .

# Django 앱 생성
PS D:\work\acorn2025Work\django3_db> python manage.py startapp myapp

# 템플릿 디렉토리 생성
PS D:\work\acorn2025Work\django3_db> mkdir templates
```

### 2. 프로젝트 디렉토리 구조

```
django3_db/
├── manage.py
├── mainapp/              # 메인 프로젝트
│   ├── __init__.py
│   ├── settings.py       # 데이터베이스 설정 포함
│   ├── urls.py          # URL 라우팅
│   ├── wsgi.py
│   └── asgi.py
├── myapp/               # 앱 디렉토리
│   ├── __init__.py
│   ├── views.py         # 뷰 함수
│   ├── models.py        # 모델 정의 (inspectdb 결과)
│   ├── admin.py
│   ├── apps.py
│   ├── tests.py
│   └── migrations/      # 마이그레이션 파일
├── templates/           # HTML 템플릿
│   ├── index.html
│   └── dbshow.html
└── aa.py               # inspectdb 결과 파일
```

---

## MariaDB 데이터베이스 설정

### 1. settings.py 데이터베이스 설정

```python
# mainapp/settings.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'mydb',
        'USER': 'root',
        'PASSWORD': '970506',
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {
           "charset": "utf8mb4",
           "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",  # 오류 방지, 데이터 형식 불일치 허용범위 초과 등
        },
    }
}

# 앱 등록 추가
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',  # 생성한 앱 추가
]

# 템플릿 설정
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # 템플릿 디렉토리 설정
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

### 2. MySQL 클라이언트 라이브러리 설치

```bash
# mysqlclient 설치 (Django에서 MySQL/MariaDB 연결용)
pip install mysqlclient

# 또는 PyMySQL 사용 시
pip install PyMySQL
```

### 3. 데이터베이스 역공학 (inspectdb)

```bash
# 기존 데이터베이스 테이블을 Django 모델로 변환
PS D:\work\acorn2025Work\django3_db> python manage.py inspectdb > aa.py
```

---

## 모델 생성 및 마이그레이션

### 1. inspectdb 결과 - aa.py

```python
# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Board(models.Model):
    num = models.IntegerField(primary_key=True)
    author = models.CharField(max_length=10, blank=True, null=True)
    title = models.CharField(max_length=50, blank=True, null=True)
    content = models.CharField(max_length=4000, blank=True, null=True)
    bwrite = models.DateField(blank=True, null=True)
    readcnt = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'board'


class Buser(models.Model):
    buserno = models.IntegerField(primary_key=True)
    busername = models.CharField(max_length=10)
    buserloc = models.CharField(max_length=10, blank=True, null=True)
    busertel = models.CharField(max_length=15, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'buser'


class Gogek(models.Model):
    gogekno = models.IntegerField(primary_key=True)
    gogekname = models.CharField(max_length=10)
    gogektel = models.CharField(max_length=20, blank=True, null=True)
    gogekjumin = models.CharField(max_length=14, blank=True, null=True)
    gogekdamsano = models.ForeignKey('Jikwon', models.DO_NOTHING, db_column='gogekdamsano', blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'gogek'


class Jikwon(models.Model):
    jikwonno = models.IntegerField(primary_key=True)
    jikwonname = models.CharField(max_length=10)
    busernum = models.IntegerField()
    jikwonjik = models.CharField(max_length=10, blank=True, null=True)
    jikwonpay = models.IntegerField(blank=True, null=True)
    jikwonibsail = models.DateField(blank=True, null=True)
    jikwongen = models.CharField(max_length=4, blank=True, null=True)
    jikwonrating = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'jikwon'


class Sangdata(models.Model):
    code = models.IntegerField(primary_key=True)
    sang = models.CharField(max_length=20, blank=True, null=True)
    su = models.IntegerField(blank=True, null=True)
    dan = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sangdata'
```

### 2. myapp/models.py에 모델 복사

`aa.py`의 내용을 `myapp/models.py`에 복사하여 사용합니다.

### 3. 마이그레이션 실행

```bash
# 마이그레이션 파일 생성 (신규 테이블 생성 시에만 필요)
PS D:\work\acorn2025Work\django3_db> python manage.py makemigrations

# 마이그레이션 적용
PS D:\work\acorn2025Work\django3_db> python manage.py migrate
```

---

## 뷰 함수 구현

### myapp/views.py

```python
from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd

def indexFunc(request):
    """메인 페이지 렌더링"""
    return render(request, 'index.html')

def dbshowFunc(request):
    """직원 정보 조회 및 통계 분석"""
    
    # 사용자로부터 부서명 받기
    dept = (request.GET.get('dept') or "").strip()

    # Inner Join SQL 쿼리
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
    
    # 부서 필터링 조건 추가
    params = []
    if dept:
        sql += " WHERE b.busername LIKE %s"
        params.append(f"%{dept}%")  # SQL 인젝션 방지

    sql += " ORDER BY j.jikwonno"  # 직원번호 기준 정렬

    # 데이터베이스 연결 및 쿼리 실행
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        # 컬럼명 추출
        cols = [c[0] for c in cursor.description]

    if rows:
        # DataFrame 생성
        df = pd.DataFrame(rows, columns=cols)
        
        # 연봉 데이터 숫자 변환 (집계 안전성을 위해)
        df['연봉'] = pd.to_numeric(df['연봉'], errors='coerce')

        # 1. 직원 목록 HTML 테이블 생성
        join_html = df[['직원번호', '직원명', '부서명', '부서전화', '직급', '연봉']].to_html(
            index=False,
            classes='table table-striped table-hover'
        )

        # 2. 직급별 연봉 통계 생성
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
        
        # NaN 값을 0으로 대체
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        
        # 통계 테이블 HTML 생성
        stats_html = stats_df.to_html(
            index=False,
            classes='table table-striped table-bordered'
        )
    else:
        join_html = "<div class='alert alert-warning'>조회된 데이터가 없습니다.</div>"
        stats_html = "<div class='alert alert-info'>통계 대상 자료가 없습니다.</div>"

    # 템플릿 컨텍스트 데이터
    context = {
        'dept': escape(dept),  # XSS 공격 방지
        'join_html': join_html,
        'stats_html': stats_html,
    }
    
    return render(request, 'dbshow.html', context)
```

---

## 템플릿 구성

### 1. templates/index.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>하나 (주) 사내 직원 정보 시스템</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2 class="mb-0">🏢 하나 (주) 사내 직원 정보</h2>
                    </div>
                    <div class="card-body text-center">
                        <p class="lead">직원 정보 조회 및 통계 분석 시스템</p>
                        <a href="/dbshow/" class="btn btn-success btn-lg">
                            📊 DB 조회 페이지로 이동
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### 2. templates/dbshow.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>직원 정보 조회 시스템</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .table-container {
            max-height: 500px;
            overflow-y: auto;
        }
        .stats-section {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <!-- 제목 -->
        <h2 class="text-center mb-4">
            📊 직원 정보 AND 직급별 연봉 관련 통계
        </h2>
        
        <!-- 검색 폼 -->
        <div class="card mb-4">
            <div class="card-body">
                <h5 class="card-title">🔍 부서별 조회</h5>
                {% if dept %}
                    <h6 class="text-muted">현재 조회: <span class="badge bg-primary">{{ dept }}</span></h6>
                {% endif %}
                
                <form method="get" action="/dbshow/" class="row g-3">
                    <div class="col-md-8">
                        <label for="dept" class="form-label">부서명:</label>
                        <input type="text" 
                               name="dept" 
                               id="dept"
                               class="form-control" 
                               value="{{ dept }}" 
                               placeholder="부서명을 입력하세요 (예: 영업부, 총무부)">
                    </div>
                    <div class="col-md-4 d-flex align-items-end">
                        <button type="submit" class="btn btn-primary me-2">🔍 조회</button>
                        <a href="/dbshow/" class="btn btn-outline-secondary">🔄 전체 보기</a>
                    </div>
                </form>
            </div>
        </div>

        <!-- 네비게이션 -->
        <div class="text-center mb-3">
            <a href="/" class="btn btn-outline-primary">🏠 메인 페이지로 이동</a>
        </div>

        <!-- 직원 목록 섹션 -->
        <div class="card mb-4">
            <div class="card-header bg-info text-white">
                <h4 class="mb-0">👥 직원 목록</h4>
            </div>
            <div class="card-body">
                <div class="table-container">
                    {{ join_html|safe }}
                </div>
            </div>
        </div>

        <!-- 통계 섹션 -->
        <div class="stats-section">
            <h4 class="text-center mb-3">📈 직급별 연봉 통계</h4>
            <div class="table-responsive">
                {{ stats_html|safe }}
            </div>
        </div>

        <!-- 하단 네비게이션 -->
        <div class="text-center mb-5">
            <a href="/" class="btn btn-primary btn-lg">🏠 메인으로 돌아가기</a>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

---

## URL 라우팅

### mainapp/urls.py

```python
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.indexFunc, name='index'),      # 메인 페이지
    path('dbshow/', views.dbshowFunc, name='dbshow'),  # DB 조회 페이지
]
```

---

## 완성된 애플리케이션

### 1. 서버 실행

```bash
# Django 개발 서버 실행
PS D:\work\acorn2025Work\django3_db> python manage.py runserver

# 브라우저에서 확인
# http://127.0.0.1:8000/         - 메인 페이지
# http://127.0.0.1:8000/dbshow/  - 직원 정보 조회 페이지
```

### 2. 주요 기능

#### 🏠 메인 페이지 (/)
- 시스템 소개
- DB 조회 페이지로 이동 버튼

#### 📊 DB 조회 페이지 (/dbshow/)
- **전체 직원 조회**: 부서명 입력 없이 조회
- **부서별 필터링**: 부서명으로 검색 (Like 검색 지원)
- **직원 목록 표시**: Inner Join으로 직원-부서 정보 결합
- **통계 분석**: 직급별 연봉 평균, 표준편차, 인원수
- **반응형 UI**: Bootstrap을 활용한 모바일 친화적 인터페이스

### 3. 데이터베이스 구조

#### 테이블 관계
```
buser (부서)     →     jikwon (직원)     →     gogek (고객)
  ↓                      ↓                      ↓
buserno ───────────── busernum          jikwonno ──── gogekdamsano
```

#### 주요 테이블
- **buser**: 부서 정보 (부서번호, 부서명, 위치, 전화번호)
- **jikwon**: 직원 정보 (직원번호, 이름, 부서번호, 직급, 연봉)
- **gogek**: 고객 정보 (고객번호, 이름, 담당자번호)
- **sangdata**: 상품 정보
- **board**: 게시판

---

## 트러블슈팅

### 1. mysqlclient 설치 오류 (Windows)

```bash
# Visual Studio Build Tools 설치 후
pip install mysqlclient

# 또는 wheel 파일 직접 설치
pip install https://download.lfd.uci.edu/pythonlibs/archived/mysqlclient-1.4.6-cp39-cp39-win_amd64.whl
```

### 2. PyMySQL 대안 사용

```python
# mainapp/__init__.py에 추가
import pymysql
pymysql.install_as_MySQLdb()
```

### 3. 한글 인코딩 문제

```python
# settings.py DATABASES 설정 확인
'OPTIONS': {
   "charset": "utf8mb4",
   "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
},
```

### 4. 마이그레이션 오류

```bash
# 기존 마이그레이션 파일 삭제 후 재실행
python manage.py makemigrations --empty myapp
python manage.py migrate
```

### 5. 정적 파일 로드 오류

```python
# settings.py 확인
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
```

### 6. inspectdb로 생성된 모델 수정

```python
# managed = False를 True로 변경하여 Django에서 관리
class Jikwon(models.Model):
    # ... 필드 정의
    
    class Meta:
        managed = True  # False에서 True로 변경
        db_table = 'jikwon'
```

---

## 직원 테이블 활용 심화 분석

### 1. inspectdb 명령어 활용

#### inspectdb란?
`inspectdb`는 Django에서 제공하는 강력한 역공학 도구입니다. 기존 데이터베이스의 테이블 구조를 분석하여 Django 모델 클래스를 자동으로 생성해줍니다.

```bash
# 원격 데이터베이스 테이블 구조를 Django 모델로 변환
python manage.py inspectdb > aa.py
```

#### 강사님 설명 포인트
> **"원격 연결은 진행해줍니다. 그럼 aa 파일이 이렇게 형성됩니다."**
> 
> **"원격 DB 내용을 불러다가 여기다 넣어 놓고 조절할 수 있어요."**

### 2. 실무 환경에서의 경쟁 상황

#### 현실적인 취업 경쟁률
강사님이 강조하신 현실적인 상황:

- **"우리 층에 지금 우리 말고... 목표가 비슷한 대리도 들어왔다는 얘기 한 30명이나"**
- **"그렇기 때문에 긴장을 해야 된다는 거예요"**
- **"엄청나게 많은 사람들이 지금도 우리와 같은 과정을 겪어가고 있다"**

### 3. 직원 정보 시스템 고도화

#### 추가 분석 기능 구현

```python
# myapp/views.py 고도화 버전

def advancedAnalysisFunc(request):
    """고급 분석 기능"""
    
    # 1. 부서별 성별 분포 분석
    gender_analysis_sql = """
    SELECT 
        b.busername AS 부서명,
        j.jikwongen AS 성별,
        COUNT(*) AS 인원수,
        ROUND(AVG(j.jikwonpay), 0) AS 평균연봉
    FROM jikwon j
    INNER JOIN buser b ON j.busernum = b.buserno
    GROUP BY b.busername, j.jikwongen
    ORDER BY b.busername, j.jikwongen
    """
    
    # 2. 직급별 연차 분석
    seniority_analysis_sql = """
    SELECT 
        j.jikwonjik AS 직급,
        COUNT(*) AS 인원수,
        ROUND(AVG(DATEDIFF(CURDATE(), j.jikwonibsail) / 365), 1) AS 평균근속년수,
        ROUND(AVG(j.jikwonpay), 0) AS 평균연봉
    FROM jikwon j
    GROUP BY j.jikwonjik
    ORDER BY 평균연봉 DESC
    """
    
    # 3. 부서별 고객 담당 현황
    customer_analysis_sql = """
    SELECT 
        b.busername AS 부서명,
        COUNT(DISTINCT j.jikwonno) AS 직원수,
        COUNT(g.gogekno) AS 담당고객수,
        ROUND(COUNT(g.gogekno) / COUNT(DISTINCT j.jikwonno), 1) AS 직원당고객수
    FROM buser b
    LEFT JOIN jikwon j ON b.buserno = j.busernum
    LEFT JOIN gogek g ON j.jikwonno = g.gogekdamsano
    GROUP BY b.busername
    ORDER BY 직원당고객수 DESC
    """
    
    with connection.cursor() as cursor:
        # 성별 분포 분석
        cursor.execute(gender_analysis_sql)
        gender_data = cursor.fetchall()
        gender_df = pd.DataFrame(gender_data, columns=['부서명', '성별', '인원수', '평균연봉'])
        
        # 직급별 분석
        cursor.execute(seniority_analysis_sql)
        seniority_data = cursor.fetchall()
        seniority_df = pd.DataFrame(seniority_data, columns=['직급', '인원수', '평균근속년수', '평균연봉'])
        
        # 고객 담당 분석
        cursor.execute(customer_analysis_sql)
        customer_data = cursor.fetchall()
        customer_df = pd.DataFrame(customer_data, columns=['부서명', '직원수', '담당고객수', '직원당고객수'])
    
    context = {
        'gender_html': gender_df.to_html(index=False, classes='table table-striped'),
        'seniority_html': seniority_df.to_html(index=False, classes='table table-striped'),
        'customer_html': customer_df.to_html(index=False, classes='table table-striped'),
    }
    
    return render(request, 'advanced_analysis.html', context)
```

### 4. 실무 팁과 강사님 조언

#### 마이그레이션 타이밍
**강사님**: *"마이그레이션은 언제하는거에요? 테이블을 만들고 마이그레이션 하고 마이그레이트 하는거에요."*

```bash
# 새로운 모델 변경사항이 있을 때마다
python manage.py makemigrations

# 변경사항을 데이터베이스에 적용
python manage.py migrate
```

#### Django ORM vs Raw SQL 선택
**강사님**: *"장고 ORM을 써도 되고 그냥 Raw SQL 문장을 써도 괜찮아요. 편한 걸로 하시면 됩니다."*

### 5. 고급 분석 템플릿

#### templates/advanced_analysis.html

```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>고급 분석 - 직원 정보 시스템</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">📊 직원 정보 고급 분석</h1>
        
        <!-- 부서별 성별 분포 -->
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h4>👥 부서별 성별 분포 및 연봉 분석</h4>
            </div>
            <div class="card-body">
                {{ gender_html|safe }}
            </div>
        </div>
        
        <!-- 직급별 근속년수 분석 -->
        <div class="card mb-4">
            <div class="card-header bg-success text-white">
                <h4>📈 직급별 근속년수 및 연봉 분석</h4>
            </div>
            <div class="card-body">
                {{ seniority_html|safe }}
            </div>
        </div>
        
        <!-- 부서별 고객 담당 현황 -->
        <div class="card mb-4">
            <div class="card-header bg-info text-white">
                <h4>🤝 부서별 고객 담당 현황</h4>
            </div>
            <div class="card-body">
                {{ customer_html|safe }}
            </div>
        </div>
        
        <!-- 네비게이션 -->
        <div class="text-center">
            <a href="/dbshow/" class="btn btn-primary me-2">기본 분석으로</a>
            <a href="/" class="btn btn-outline-primary">메인으로</a>
        </div>
    </div>
</body>
</html>
```

### 6. 강사님의 실무 개발 패턴과 핵심 포인트

#### 마이그레이션 타이밍과 패턴 인식
**강사님**: *"마이그레이션 언제 하는 거야? 테이블을 만들어 두고... 그때마다 메이크 마이그레이션 해줘야 돼."*

```bash
# 테이블 생성 후 마이그레이션
python manage.py makemigrations

# 테이블 수정 후에도 마이그레이션
python manage.py makemigrations

# 마이그레이션 적용
python manage.py migrate
```

#### 패턴 읽기의 중요성
**강사님**: *"패턴을 안 읽고 그냥 맹목적으로... 패턴을 읽어야 돼요. 명령을 외우라는 얘기가 아니야 패턴을 읽으세요."*

### 7. Pandas DataFrame 통계 분석 구현

#### 직급별 연봉 통계 분석
```python
# myapp/views.py 고도화 - DataFrame 통계 처리

def dbshowFunc(request):
    # ... 기존 코드 ...
    
    if rows:
        df = pd.DataFrame(rows, columns=cols)
        df['연봉'] = pd.to_numeric(df['연봉'], errors='coerce')
        
        # 직급별 연봉 통계 (NaN 처리 포함)
        stats_df = (
            df.groupby('직급')['연봉']
              .agg(
                  평균='mean',
                  표준편차=lambda x: x.std(ddof=0),  # 자유도 0으로 설정
                  인원수='count',
              )
              .round(2)
              .reset_index()
              .sort_values(by='평균', ascending=False)
        )
        
        # NaN 값을 0으로 대체 (1명인 경우 표준편차가 없음)
        stats_df['표준편차'] = stats_df['표준편차'].fillna(0)
        
        # HTML 테이블 생성
        join_html = df[['직원번호', '직원명', '부서명', '부서전화', '직급', '연봉']].to_html(
            index=False,
            classes='table table-striped table-hover'
        )
        
        stats_html = stats_df.to_html(
            index=False,
            classes='table table-striped table-bordered'
        )
    else:
        join_html = "조회된 데이터가 없습니다."
        stats_html = "통계 대상 자료가 없습니다."
    
    # XSS 공격 방지를 위한 escape 처리
    context = {
        'dept': escape(dept),  # 특수문자를 HTML 엔티티로 변환
        'join_html': join_html,
        'stats_html': stats_html,
    }
    
    return render(request, 'dbshow.html', context)
```

### 8. 웹 보안 고려사항

#### SQL 인젝션 방지
**강사님**: *"문자를 더하기 하면 안 돼... SQL 인젝션 해킹에 걸려요"*

```python
# ❌ 위험한 방법 - SQL 인젝션 취약
sql = f"SELECT * FROM jikwon WHERE busername = '{dept}'"

# ✅ 안전한 방법 - 매개변수화된 쿼리
sql += " WHERE b.busername LIKE %s"
params.append(f"%{dept}%")
cursor.execute(sql, params)
```

#### XSS 공격 방지
**강사님**: *"크로스 사이트 스크립팅... 자바스크립트 내용이 해당 컴퓨터로 가가지고 다른 작업을 해"*

```python
from django.utils.html import escape

# 사용자 입력 데이터 이스케이프 처리
context = {
    'dept': escape(dept),  # <script> → &lt;script&gt;
}
```

```html
<!-- 템플릿에서 안전한 HTML 렌더링 -->
{{ join_html|safe }}  <!-- 신뢰할 수 있는 HTML만 safe 필터 사용 -->
```

### 9. 실무 개발 프로세스

#### URL 라우팅 패턴
**강사님**: *"디비쇼 요청이 들어오면 디비쇼 FNC로 가는데..."*

```python
# mainapp/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.indexFunc, name='index'),
    path('dbshow/', views.dbshowFunc, name='dbshow'),  # 슬래시 주의
]
```

#### 템플릿 개발 패턴
```html
<!-- templates/dbshow.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>직원 정보 AND 직급별 연봉 관련 통계</title>
</head>
<body>
    <h2>직원 정보 AND 직급별 연봉 관련 통계</h2>
    <h3>부서명: {{ dept }}</h3>
    
    <!-- 검색 폼 -->
    <form method="get" action="/dbshow/">
        <label for="dept">부서명:</label>
        <input type="text" name="dept" value="{{ dept }}" 
               placeholder="부서명 입력 (예: 총무부, 영업부)">
        <button type="submit">조회</button>
    </form>
    
    <a href="/dbshow/">전체 자료</a>
    <a href="/">메인 페이지로 이동</a>
    
    <h3>직원 목록</h3>
    {{ join_html|safe }}
    
    <h3>직급별 연봉 통계</h3>
    {{ stats_html|safe }}
</body>
</html>
```

### 10. 팀 프로젝트 대비 실무 팁

#### 강사님의 프로젝트 지침
**강사님**: *"팀별 프로젝트 짤 때는 꼭 위임을 해줘야 돼요... 랜덤하게 찍어가지고 팀원 중에 누군가가 발표"*

#### 보안 코딩 가이드라인
**강사님**: *"시큐어 코딩 가이드라인... 면접 같은 데 가가지고 '시큐어 코딩 가이드라인에 대해서 알고 있나요?' 이런 얘기할 때"*

```python
# 보안 코딩 체크리스트
# 1. SQL 인젝션 방지 - 매개변수화된 쿼리 사용
# 2. XSS 방지 - 사용자 입력 이스케이프 처리
# 3. CSRF 방지 - Django의 기본 CSRF 토큰 활용
# 4. 입력 검증 - 적절한 데이터 타입 및 범위 검증
# 5. 예외 처리 - 민감한 정보 노출 방지
```

### 11. 실습 문제 및 평가 기준

#### 강사님의 평가 방식
**강사님**: *"문제를 복붙해가지고 내가 직접 돌려본다. 안 돌아가면 0점이야!"*

#### 실습 문제 예시
```
문제 8: MariaDB의 직원 분석 테이블을 이용하여 다음 작업을 실시하시오.

1. 사원 지역, 부서, 직급, 근무 연수를 포함하여 조회
2. 사번, 이름, 부서, 직급, 성별, 연봉을 조회
3. 부서별 평균 연봉 막대그래프 출력
4. 성별, 직급별 빈도표 출력
```

### 12. Django ORM vs Raw SQL 활용

#### 강사님의 조언
**강사님**: *"장고 ORM을 써도 되고 그냥 Raw SQL 문장을 써도 괜찮아요. 편한 걸로 하시면 됩니다."*

```python
# Raw SQL 방식 (현재 가이드에서 사용)
with connection.cursor() as cursor:
    cursor.execute(sql, params)
    rows = cursor.fetchall()

# Django ORM 방식 (대안)
from .models import Jikwon, Buser

jikwons = Jikwon.objects.select_related('busernum').all()
if dept:
    jikwons = jikwons.filter(busernum__busername__icontains=dept)
```

### 13. 데이터베이스 안전 관리

#### 강사님의 당부
**강사님**: *"DB는 정말 어린아이 다루듯이 다뤄야 돼... 꼭 백업을 받아놔야 돼요"*

```bash
# MariaDB 백업
mysqldump -u root -p mydb > backup_$(date +%Y%m%d).sql

# MariaDB 복원
mysql -u root -p mydb < backup_20250812.sql

# 안전한 종료
mysql> exit;  # 반드시 정상 종료
```

### 14. 실무에서의 AI 트렌드 반영

#### 강사님의 미래 전망
**강사님**: *"요즘은 뭐 거의 뉴스가요 AI 뉴스인 중국과입니다... 양자컴퓨터... 나의 시대는 그게 별 권력이 없겠지만 여름 시대는요. 이제 분명히 이제 고녀석이거든요."*

#### AI 기능 통합 아이디어
```python
# 향후 AI 기능 통합 예시
def aiAnalysisFunc(request):
    """AI 기반 직원 분석 (향후 구현)"""
    
    # 1. 직원 이직 예측 모델
    # 2. 최적 부서 배치 추천
    # 3. 연봉 적정성 분석
    # 4. 고객 만족도 예측
    
    pass
```

---

## 추가 학습 포인트

### 1. Django ORM vs Raw SQL
- **ORM 장점**: 안전성, 가독성, 유지보수성
- **Raw SQL 장점**: 복잡한 쿼리, 성능 최적화
- **강사님 조언**: "편한 걸로 그때그때 상황에 맞춰서 하면은 그죠?"

### 2. 보안 고려사항
- **SQL 인젝션 방지**: 매개변수화된 쿼리 사용
- **XSS 방지**: `escape()` 함수로 사용자 입력 처리
- **CSRF 방지**: Django의 기본 CSRF 보호 기능 활용

### 3. 성능 최적화
- **데이터베이스 인덱스**: 자주 조회되는 컬럼에 인덱스 생성
- **쿼리 최적화**: N+1 문제 해결, 적절한 Join 사용
- **캐싱**: 반복적인 쿼리 결과 캐싱

### 4. 확장 아이디어
- **페이지네이션**: 대용량 데이터 처리
- **검색 기능**: 전문 검색 엔진 연동
- **차트 시각화**: ECharts, Chart.js 연동
- **Excel 내보내기**: 분석 결과 다운로드
- **REST API**: 모바일 앱 연동
- **AI 기능**: 예측 분석, 추천 시스템

### 5. 강사님의 격려 메시지
> **"초심을 잃으면 안 되겠다... 더해야 돼. 더해야 돼."**
> 
> **"나만 하는 거 아니라는 거지 많은 사람들이 하고 있어요. 그거를 그다음에 즐겁게 받아들이고 이겨내야 되겠지."**

이제 Django와 MariaDB를 연결하여 실무에서 활용할 수 있는 데이터베이스 연동 웹 애플리케이션을 완성했습니다! 경쟁이 치열한 현실 속에서도 꾸준히 실력을 쌓아 나가면 반드시 성공할 수 있습니다! 🚀