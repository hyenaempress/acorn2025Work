# 20.03 Django MariaDB 연결 및 DB 활용 완전 가이드

## 📋 목차
1. [프로젝트 생성 및 기본 설정](#프로젝트-생성-및-기본-설정)
2. [MariaDB 데이터베이스 설정](#mariadb-데이터베이스-설정)
3. [inspectdb를 통한 모델 생성](#inspectdb를-통한-모델-생성)
4. [URL 라우팅 설정](#url-라우팅-설정)
5. [뷰 함수 구현](#뷰-함수-구현)
6. [템플릿 구성](#템플릿-구성)
7. [실행 및 테스트](#실행-및-테스트)
8. [완성된 프로젝트 구조](#완성된-프로젝트-구조)

---

## 프로젝트 생성 및 기본 설정

### 1. 프로젝트 및 앱 생성

```bash
# 1. 프로젝트 디렉토리 생성
PS D:\work\acorn2025Work> mkdir django3_db

# 2. Django 프로젝트 생성
PS D:\work\acorn2025Work> cd django3_db
PS D:\work\acorn2025Work\django3_db> django-admin startproject mainapp .

# 3. Django 앱 생성
PS D:\work\acorn2025Work\django3_db> python manage.py startapp myapp
```

### 2. 기본 디렉토리 구조

```
django3_db/
├── manage.py
├── mainapp/                    # 메인 프로젝트 설정
│   ├── __init__.py
│   ├── settings.py            # 데이터베이스 설정
│   ├── urls.py               # URL 라우팅
│   ├── wsgi.py
│   └── asgi.py
├── myapp/                     # 앱 디렉토리
│   ├── __init__.py
│   ├── views.py              # 뷰 함수
│   ├── models.py             # 모델 (inspectdb 결과)
│   ├── admin.py
│   ├── apps.py
│   ├── tests.py
│   └── migrations/
├── templates/                 # 템플릿 디렉토리
│   ├── index.html
│   └── dbshow.html
├── static/                    # 정적 파일
└── aa.py                     # inspectdb 출력 파일
```

---

## MariaDB 데이터베이스 설정

### 1. settings.py 설정 변경

**여기서 데이터베이스를 바꿔줍니다**

```python
# mainapp/settings.py

# 나머지는 위의 파일 기본 세팅이랑 같음
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
           "init_command": "SET sql_mode='STRICT_TRANS_TABLES'", # 오류 방지, 데이터 형식 불일치 허용범위 초과 등
        },
    }
}

# 앱 등록
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

### 2. 필수 라이브러리 설치

```bash
# MySQL 클라이언트 설치
pip install mysqlclient

# 또는 PyMySQL 사용 시
pip install PyMySQL
```

---

## inspectdb를 통한 모델 생성

### 1. inspectdb 실행

```bash
# 이걸로 원격 연결은 진행해줍니다
PS D:\work\acorn2025Work\django3_db> python manage.py inspectdb > aa.py
```

**inspectdb란 무엇일까?**
> 기존 데이터베이스의 테이블 구조를 분석하여 Django 모델 클래스를 자동으로 생성해주는 명령어입니다.

### 2. 생성된 모델 파일 (aa.py)

**그럼 aa 파일이 이렇게 형성됩니다**

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

### 3. 마이그레이션 실행

**마이그레이션은 언제하는거에요? 테이블을 만들고 마이그레이션 하고 마이그레이트 하는거에요. 그때마다 메이크 마이그레이션을 해줘서 쓰면 됩니다.**

```bash
# 여기서 python manage.py migrate 로 마이그레이션
PS D:\work\acorn2025Work\django3_db> python manage.py migrate
```

---

## URL 라우팅 설정

### 1. 메인 URL 설정

**이제 라우팅을 해봐야 합니다**

```python
# mainapp/urls.py
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.indexFunc, name='index'),
    path('dbshow/', views.dbshowFunc, name='dbshow'),
]
```

---

## 뷰 함수 구현

### 1. 기본 뷰 구조

**추가 - 뷰스에서 연결**

```python
# myapp/views.py 초기 버전
from django.shortcuts import render

def indexFunc(request):
    return render(request, 'index.html')

def dbshowFunc(request):
    pass
```

### 2. 완전한 뷰 함수 구현

**이제 다시 뷰로가서 작업 합니다**

```python
# myapp/views.py 완전 버전
from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd

def indexFunc(request):
    return render(request, 'index.html')

def dbshowFunc(request):
    # 사용자로부터 부서명을 받는다
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
        params.append(f"%{dept}%")  # SQL 해킹 방지

    sql += " ORDER BY j.jikwonno"  # 직원번호 기준 정렬

    # 데이터베이스 연결 및 쿼리 실행
    with connection.cursor() as cursor:
        cursor.execute(sql, params)  # 파람스는 퍼센트랑 매핑합니다
        rows = cursor.fetchall()
        # 커서들은 쿼리 정보에 대한 메타데이터를 가지고 있습니다
        cols = [c[0] for c in cursor.description]

    if rows:
        # DataFrame 생성 및 처리
        df = pd.DataFrame(rows, columns=cols)
        # 집계 안전을 위해 숫자 변환
        df['연봉'] = pd.to_numeric(df['연봉'], errors='coerce')

        # 조인 결과로 HTML 테이블 생성이 필요합니다
        join_html = df[['직원번호', '직원명', '부서명', '부서전화', '직급', '연봉']].to_html(index=False)

        # 직급별 연봉 통계표 (NaN -> 0)
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

    # 문자열에 특수문자가 있는 경우 HTML 엔티티로 취급함 XSS 공격을 막기 위한 것
    ctx = {
        'dept': escape(dept),
        'join_html': join_html,
        'stats_html': stats_html,
    }
    
    return render(request, 'dbshow.html', ctx)
```

---

## 템플릿 구성

### 1. 메인 페이지 템플릿

**우선은 인덱스부터 연결해줍니다**

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>하나 (주) 사내 직원 정보 시스템</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; text-align: center; }
        .btn { 
            display: inline-block; 
            padding: 10px 20px; 
            background-color: #007bff; 
            color: white; 
            text-decoration: none; 
            border-radius: 5px; 
            margin: 10px;
        }
        .btn:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏢 하나 (주) 사내 직원 정보</h2>
        <p>직원 정보 조회 및 통계 분석 시스템</p>
        <a href="/dbshow/" class="btn">📊 DB 조회 페이지로 이동</a>
    </div>
</body>
</html>

<!-- 이건 순수 HTML과 다릅니다 -->
```

### 2. DB 조회 페이지 템플릿

**이번엔 또 템플릿 업데이트**

```html
<!-- templates/dbshow.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>직원 정보 조회 시스템</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        .form-group label { display: inline-block; width: 100px; }
        .form-group input { padding: 8px; width: 200px; }
        .btn { 
            padding: 8px 15px; 
            background-color: #007bff; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover { background-color: #0056b3; }
        .nav-links a { 
            display: inline-block; 
            margin: 10px; 
            color: #007bff; 
            text-decoration: none;
        }
        .nav-links a:hover { text-decoration: underline; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0;
        }
        table, th, td { 
            border: 1px solid #ddd; 
        }
        th, td { 
            padding: 8px; 
            text-align: left; 
        }
        th { 
            background-color: #f2f2f2; 
        }
        .section { 
            margin: 30px 0; 
            padding: 20px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>📊 직원 정보 AND 직급별 연봉 관련 통계</h2>
        <h3>🔍 부서명: {{ dept }}</h3>
        
        <!-- 검색 폼 -->
        <div class="section">
            <form method="get" action="/dbshow/">
                <div class="form-group">
                    <label for="dept">부서명:</label>
                    <input type="text" name="dept" value="{{ dept }}" 
                           placeholder="부서명 입력 (예: 총무부, 영업부)">
                    <button type="submit" class="btn">🔍 조회</button>
                </div>
            </form>
            
            <div class="nav-links">
                <a href="/dbshow/">📋 전체 자료</a>
                <a href="/">🏠 메인 페이지로 이동</a>
            </div>
        </div>

        <!-- 직원 목록 -->
        <div class="section">
            <h3>👥 직원 목록</h3>
            {{ join_html|safe }}
        </div>

        <!-- 통계 정보 -->
        <div class="section">
            <h3>📈 직급별 연봉 통계</h3>
            {{ stats_html|safe }}
        </div>
    </div>
</body>
</html>
```

---

## 실행 및 테스트

### 1. 서버 실행

```bash
# python manage.py runserver 로 확인
PS D:\work\acorn2025Work\django3_db> python manage.py runserver
```

**홈페이지가 이제 잘 나옵니다**

### 2. 테스트 방법

1. **메인 페이지 접속**: `http://127.0.0.1:8000/`
2. **전체 직원 조회**: `http://127.0.0.1:8000/dbshow/`
3. **부서별 조회**: 
   - 총무부: 폼에서 "총무부" 입력
   - 영업부: 폼에서 "영업부" 입력
   - 전산부: 폼에서 "전산부" 입력
   - 관리부: 폼에서 "관리부" 입력

---

## 완성된 프로젝트 구조

```
django3_db/
├── 📁 mainapp/                 # Django 프로젝트 설정
│   ├── 📄 __init__.py
│   ├── 📄 asgi.py
│   ├── 📄 settings.py          # ✅ MariaDB 설정
│   ├── 📄 urls.py              # ✅ URL 라우팅
│   └── 📄 wsgi.py
├── 📁 myapp/                   # Django 앱
│   ├── 📁 __pycache__/
│   ├── 📁 migrations/
│   ├── 📄 __init__.py
│   ├── 📄 admin.py
│   ├── 📄 apps.py
│   ├── 📄 models.py            # (inspectdb 결과 복사)
│   ├── 📄 tests.py
│   └── 📄 views.py             # ✅ 뷰 함수 구현
├── 📁 static/                  # 정적 파일 (CSS, JS, 이미지)
├── 📁 templates/               # 템플릿 디렉토리
│   ├── 📄 dbshow.html          # ✅ DB 조회 페이지
│   └── 📄 index.html           # ✅ 메인 페이지
├── 📄 aa.py                    # inspectdb 출력 파일
└── 📄 manage.py                # Django 관리 스크립트
```

---

## 참고 DB 데이터

**아래 자료는 실습을 위해 가상으로 만들어진 자료임을 밝힙니다**

### 테이블 생성 및 데이터 삽입

```sql
-- 상품 데이터
create table sangdata(
code int primary key,
sang varchar(20),
su int,
dan int);                    -- 참고: 한글이 깨질 경우 ... dan int)charset=utf8;

insert into sangdata values(1,'장갑',3,10000);
insert into sangdata values(2,'벙어리장갑',2,12000);
insert into sangdata values(3,'가죽장갑',10,50000);
insert into sangdata values(4,'가죽점퍼',5,650000);

-- 부서 데이터
create table buser(
buserno int primary key, 
busername varchar(10) not null,
buserloc varchar(10),
busertel varchar(15));

insert into buser values(10,'총무부','서울','02-100-1111');
insert into buser values(20,'영업부','서울','02-100-2222');
insert into buser values(30,'전산부','서울','02-100-3333');
insert into buser values(40,'관리부','인천','032-200-4444');

-- 직원 데이터
create table jikwon(
jikwonno int primary key,
jikwonname varchar(10) not null,
busernum int not null,
jikwonjik varchar(10) default '사원', 
jikwonpay int,
jikwonibsail date,
jikwongen varchar(4),
jikwonrating char(3),
CONSTRAINT ck_jikwongen check(jikwongen='남' or jikwongen='여'));

insert into jikwon values(1,'홍길동',10,'이사',9900,'2008-09-01','남','a');
insert into jikwon values(2,'한송이',20,'부장',8800,'2010-01-03','여','b');
insert into jikwon values(3,'이순신',20,'과장',7900,'2010-03-03','남','b');
-- ... (30명의 직원 데이터)

-- 고객 데이터
create table gogek(
gogekno int primary key,
gogekname varchar(10) not null,
gogektel varchar(20),
gogekjumin char(14),
gogekdamsano int,
CONSTRAINT FK_gogekdamsano foreign key(gogekdamsano) references jikwon(jikwonno));

insert into gogek values(1,'이나라','02-535-2580','850612-1156777',5);
-- ... (15명의 고객 데이터)

-- 게시판 데이터
create table board(
num int primary key,
author varchar(10),
title varchar(50),
content varchar(4000),
bwrite date,
readcnt int default 0);

insert into board(num,author,title,content,bwrite) values(1,'홍길동','연습','연습내용',now());
```

---

## 주요 기능

### ✅ 완성된 기능들

1. **MariaDB 연결**: Django와 MariaDB 완전 연동
2. **부서별 검색**: 동적 검색 기능 구현
3. **Inner Join**: 직원-부서 정보 결합 조회
4. **통계 분석**: Pandas를 활용한 직급별 연봉 통계
5. **반응형 UI**: 깔끔한 웹 인터페이스
6. **보안 처리**: SQL 인젝션 및 XSS 공격 방지

### 🎯 핵심 학습 포인트

- **inspectdb 활용**: 기존 DB를 Django 모델로 변환
- **Raw SQL 사용**: Django ORM 대신 직접 SQL 쿼리 실행
- **Pandas 연동**: DataFrame을 활용한 데이터 분석
- **템플릿 엔진**: Django 템플릿 시스템 활용
- **보안 코딩**: 안전한 웹 애플리케이션 개발

이제 Django와 MariaDB를 연결하여 실무에서 활용할 수 있는 완전한 직원 정보 관리 시스템이 완성되었습니다! 🚀