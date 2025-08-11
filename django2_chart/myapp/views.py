from django.shortcuts import render
import json, os
import pandas as pd
import numpy as np
import requests
from django.conf import settings
from datetime import datetime


DATA_DIR = os.path.join(settings.BASE_DIR, 'data') #데이터 폴더가 없으면 작업을 안합니다다
CSV_PATH = os.path.join(DATA_DIR, 'seattle_weather.csv')
CSV_URL = "https://raw.githubusercontent.com/vega/vega-datasets/master/data/seattle-weather.csv"

# Create your views here.
def index(request):
    return render(request, 'index.html')

def csvFunc():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        res = requests.get(CSV_URL, timeout=20) # 20초 동안 데이터를 받지 못하면 오류 발생
        res.raise_for_status() # 오류 발생 시 오류 메시지 출력
        # 잘 들어왔다고 한다면 저장에 참여합니다 
        with open(CSV_PATH, mode= 'wb') as f:
            f.write(res.content) #바이너리 형식으로 잃습니다. 빼버리고 text 로 읽을 수도 있지만 한글이 깨질 수 있어요 
        
def show(request):
    csvFunc() #데이터가 확보가 된 상황 
    df = pd.read_csv(CSV_PATH)
    print(df.columns) #['date', 'temp_max', 'temp_min', 'temp_mean', 'precipitation', 'wind', 'weather']
    print(df.info())
    
    #일부 열만 작업에 참여 시킵니다.
    df = df[['date', 'precipitation', 'temp_max', 'temp_min', 'wind']].copy() #원본은 유지한 채 새로운 테이블이 만들어짐 

    df['date'] = pd.to_datetime(df['date']) #타입이 바뀌어서 날짜 연산이 가능해진다.
    df = df.dropna() #결측치 제거 
    
    #기술 통계 - 평균 / 표준편차 ... 등등등 
    
    stats_df = df[['precipitation', 'temp_max', 'temp_min']].describe().round(3) #요약 통계를 구해줍니다.
    print('stats_df: \n', stats_df) 
    
    #df 의 상위 5행 출력에 참여시킨다 
    head_html = df.head(5).to_html(classes= 'table table-sm table-striped', index=False, border=0)
    stats_html = stats_df.to_html(classes= 'table table-sm table-striped', border=0)
    
    #Echarts 용 데이터 (월별 평균 , 최고기온 , 최저기온 )
    #월 단위 평균 최고 기온 집계 
    monthly = (
        df.set_index('date')
        .resample('ME')[['temp_max']] #리셈플링 일정 기간 단위로 집계 합니다 
        .mean() #평균 값을 구합니다.
        .reset_index() #인덱스를 컬럼으로 변환 

    )
   #print('monthly: \n', monthly.head(2))
    
    #2012-01-31  7.054839 -> 2012-01 7.05 로 변환
    labels = monthly['date'].dt.strftime('%Y-%m').tolist()
       #  print('labels: \n', labels)
    series = monthly['temp_max'].round(2).tolist()
      #  print('series: \n', series)
                    
    ctx_dict = {
        'head_html': head_html,
        'stats_html': stats_html,
        'labels_json': json.dumps(labels, ensure_ascii=False), 
        'series_json': json.dumps(series, ensure_ascii=False)
    }
    
    return render(request, 'show.html', ctx_dict) #이걸로 
