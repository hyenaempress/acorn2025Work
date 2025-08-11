from django.shortcuts import render
import json, os
import pandas as pd
import numpy as np
import requests
from django.conf import settings
from datetime import datetime


DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
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

def show(request):
    
    pass

