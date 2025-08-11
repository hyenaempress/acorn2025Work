from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로 그리기
import matplotlib.pyplot as plt
import pandas as pd

def main(request):
    return render(request, 'main.html')

def showdata(request):
    # 데이터 로딩 (seaborn 실패 시 sklearn으로 대체)
    try:
        import seaborn as sns
        df = sns.load_dataset('iris')
    except Exception:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame.rename(columns={
            'sepal length (cm)': 'sepal_length',
            'sepal width (cm)': 'sepal_width',
            'petal length (cm)': 'petal_length',
            'petal width (cm)': 'petal_width'
        })
        df['species'] = df['target'].map(dict(enumerate(iris.target_names)))
        df = df.drop(columns=['target'])

    # 이미지 저장 경로 준비: <BASE_DIR>/static/images/iris.png
    static_img_dir = Path(settings.BASE_DIR) / 'static' / 'images'
    static_img_dir.mkdir(parents=True, exist_ok=True)
    img_path = static_img_dir / 'iris.png'

    # 파이차트 그려서 파일 저장
    counts = df['species'].value_counts().sort_index()
    plt.figure()
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ylabel='')
    plt.title('iris species counts')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()

    # 테이블 HTML 생성 (여기서 옵션 다 줘서 '문자열'로 완성)
    table_html = df.to_html(classes='table table-striped table-sm', index=False)

    return render(request, 'showdata.html', {
        'table': table_html,           # ← 문자열 그대로 전달 (호출 X)
        'img_relpath': 'images/iris.png'
    })


