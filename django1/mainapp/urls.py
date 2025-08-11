"""
URL configuration for mainapp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views # 장고 앱 추가 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main, name='main'), #위임 하는게 좋아요  메인 쓰고 있지만 , 요청명이 없을땐 views.main 쓰면 됩니다.   
    path('showdata', views.showdata, name='showdata'), #위임 하는게 좋아요  메인 쓰고 있지만 , 요청명이 없을땐 views.main 쓰면 됩니다.    
]

