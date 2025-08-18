from django.urls import path
from . import views   # ✅ 같은 앱의 views를 상대 임포트

urlpatterns = [
    path('', views.survey_main, name='index'),
    path('coffee/survey', views.survey_form, name='survey_form'),
    path('coffee/surveyshow', views.survey_result, name='survey_result'),
    path('coffee/serveyprocess', views.survey_process, name='survey_process'),
    
    
]

