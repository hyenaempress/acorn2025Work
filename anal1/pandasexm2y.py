
"""
Pandas 문제 5, 6번 - 타이타닉 데이터 및 기타 데이터 분석
작성자: 박수연
작성일: 2025-08-06
"""

# ==========================================
# 1. 라이브러리 임포트
# ==========================================
import pandas as pd
import numpy as np

# ==========================================
# 2. 문제 5번 - 타이타닉 데이터 분석
# ==========================================

# 타이타닉 데이터 URL
TITANIC_URL = ('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
# 타이타닉 데이터 로드
df = pd.read_csv(TITANIC_URL) # 타이타닉 데이터 로드 CSV 파일 열기 인터넷에서 가져왔음 
print(df.head(3))
# 데이터 기본 정보 확인
print(df.info())
#해당 파일을 CSV 파일로 저장
df.to_csv('titanic_data.csv', index=False)



