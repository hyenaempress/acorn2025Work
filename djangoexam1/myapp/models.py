from django.db import models

# Create your models here.
# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.

# ---- 기본 테이블들 ----


class Buser(models.Model):
    buserno = models.IntegerField(primary_key=True)
    busername = models.CharField(max_length=10)
    buserloc = models.CharField(max_length=10, blank=True, null=True)
    busertel = models.CharField(max_length=15, blank=True, null=True)

    class Meta:
        managed = False   # 기존 DB 사용만 할 거면 False가 안전
        db_table = 'buser'


class Jikwon(models.Model):
    jikwonno = models.IntegerField(primary_key=True)
    jikwonname = models.CharField(max_length=10)
    # IntegerField -> ForeignKey로 교체 (관계 필드)
    busernum = models.ForeignKey(
        Buser,
        on_delete=models.PROTECT,     # 필요에 따라 CASCADE/DO_NOTHING 등으로 조정
        db_column='busernum'
    )
    jikwonjik = models.CharField(max_length=10, blank=True, null=True)
    jikwonpay = models.IntegerField(blank=True, null=True)
    jikwonibsail = models.DateField(blank=True, null=True)
    jikwongen = models.CharField(max_length=4, blank=True, null=True)
    jikwonrating = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'jikwon'


class Gogek(models.Model):
    gogekno = models.IntegerField(primary_key=True)
    gogekname = models.CharField(max_length=10)
    gogektel = models.CharField(max_length=20, blank=True, null=True)
    gogekjumin = models.CharField(max_length=14, blank=True, null=True)
    gogekdamsano = models.ForeignKey(
        Jikwon,
        on_delete=models.DO_NOTHING,
        db_column='gogekdamsano',
        blank=True,
        null=True
    )

    class Meta:
        managed = False
        db_table = 'gogek'


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


class Sangdata(models.Model):
    code = models.IntegerField(primary_key=True)
    sang = models.CharField(max_length=20, blank=True, null=True)
    su = models.IntegerField(blank=True, null=True)
    dan = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'sangdata'
