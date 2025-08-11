from django.shortcuts import render

# Create your views here.

def main(request):
    return render(request, 'main.html') #랜더링할 대상이 누구인지 넣어줘야 합니다.

def showdata(request):
    pass
