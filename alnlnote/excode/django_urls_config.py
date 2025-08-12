# mainapp/urls.py - URL 라우팅 설정

from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # 메인 페이지
    path('', views.indexFunc, name='index'),
    
    # Raw SQL 버전 (기존)
    path('dbshow/', views.dbshowFunc, name='dbshow'),
    
    # Django ORM 버전 (신규 추가)
    path('orm-analysis/', views.ormAnalysisFunc, name='orm_analysis'),
    
    # ORM 기본 예제
    path('orm-examples/', views.ormBasicExamples, name='orm_examples'),
    
    # 고급 분석 (기존)
    path('advanced-analysis/', views.advancedAnalysisFunc, name='advanced_analysis'),
]

# myapp/models.py에 추가해야 할 모델 수정사항

"""
Django ORM을 완전히 활용하려면 models.py에서 다음 수정이 필요합니다:

1. managed = False → True로 변경
2. ForeignKey 관계 설정
3. __str__ 메서드 추가

아래는 수정된 models.py 예시입니다:
"""

# myapp/models.py (수정 버전)
from django.db import models

class Buser(models.Model):
    buserno = models.IntegerField(primary_key=True)
    busername = models.CharField(max_length=10)
    buserloc = models.CharField(max_length=10, blank=True, null=True)
    busertel = models.CharField(max_length=15, blank=True, null=True)

    class Meta:
        managed = True  # False에서 True로 변경
        db_table = 'buser'
        
    def __str__(self):
        return f"{self.busername} ({self.buserno})"

class Jikwon(models.Model):
    jikwonno = models.IntegerField(primary_key=True)
    jikwonname = models.CharField(max_length=10)
    # ForeignKey로 관계 설정 (기존: busernum = models.IntegerField())
    busernum = models.ForeignKey(
        Buser, 
        on_delete=models.CASCADE, 
        db_column='busernum',
        related_name='jikwon'  # 역참조 이름 설정
    )
    jikwonjik = models.CharField(max_length=10, blank=True, null=True)
    jikwonpay = models.IntegerField(blank=True, null=True)
    jikwonibsail = models.DateField(blank=True, null=True)
    jikwongen = models.CharField(max_length=4, blank=True, null=True)
    jikwonrating = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = True  # False에서 True로 변경
        db_table = 'jikwon'
        
    def __str__(self):
        return f"{self.jikwonname} ({self.jikwonjik})"

class Gogek(models.Model):
    gogekno = models.IntegerField(primary_key=True)
    gogekname = models.CharField(max_length=10)
    gogektel = models.CharField(max_length=20, blank=True, null=True)
    gogekjumin = models.CharField(max_length=14, blank=True, null=True)
    # ForeignKey 관계 명확화
    gogekdamsano = models.ForeignKey(
        Jikwon, 
        on_delete=models.SET_NULL,  # CASCADE 대신 SET_NULL 권장
        db_column='gogekdamsano', 
        blank=True, 
        null=True,
        related_name='gogek_customers'
    )

    class Meta:
        managed = True  # False에서 True로 변경
        db_table = 'gogek'
        
    def __str__(self):
        return f"{self.gogekname} ({self.gogekno})"

class Board(models.Model):
    num = models.IntegerField(primary_key=True)
    author = models.CharField(max_length=10, blank=True, null=True)
    title = models.CharField(max_length=50, blank=True, null=True)
    content = models.CharField(max_length=4000, blank=True, null=True)
    bwrite = models.DateField(blank=True, null=True)
    readcnt = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = True  # False에서 True로 변경
        db_table = 'board'
        
    def __str__(self):
        return f"{self.title} - {self.author}"

class Sangdata(models.Model):
    code = models.IntegerField(primary_key=True)
    sang = models.CharField(max_length=20, blank=True, null=True)
    su = models.IntegerField(blank=True, null=True)
    dan = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = True  # False에서 True로 변경
        db_table = 'sangdata'
        
    def __str__(self):
        return f"{self.sang} ({self.code})"

"""
📝 모델 수정 후 마이그레이션 실행:

1. 마이그레이션 파일 생성:
   python manage.py makemigrations myapp

2. 마이그레이션 적용:
   python manage.py migrate

3. 만약 오류가 발생하면:
   python manage.py makemigrations --empty myapp
   python manage.py migrate

⚠️ 주의사항:
- 기존 데이터가 있는 상태에서 ForeignKey로 변경할 때는 주의 필요
- 데이터 무결성 확인 후 마이그레이션 실행
- 백업 후 작업 권장
"""