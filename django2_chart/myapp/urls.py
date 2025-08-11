
from django.urls import path, include
from myapp import views
from myapp import urls

urlpatterns = [
    path('', views.show, name='show'),

]


