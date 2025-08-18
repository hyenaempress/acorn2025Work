from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.db.models import Count      # ✅ 추가
from .models import Survey

def survey_main(request):
    return render(request, 'index.html')

def survey_form(request):
    return render(request, 'coffee/survey_form.html')

def survey_result(request):
    surveys = Survey.objects.all()

    agg = Survey.objects.values('gender', 'co_survey').annotate(cnt=Count('rnum'))
    genders = sorted({row['gender'] for row in agg})
    brands  = sorted({row['co_survey'] for row in agg})

    result_text = "데이터가 부족합니다."
    chi2 = dof = None
    decision = None

    if len(genders) >= 2 and len(brands) >= 2:
        idx_g = {g:i for i,g in enumerate(genders)}
        idx_b = {b:j for j,b in enumerate(brands)}
        r, c = len(genders), len(brands)
        obs = [[0 for _ in range(c)] for __ in range(r)]
        for row in agg:
            obs[idx_g[row['gender']]][idx_b[row['co_survey']]] = row['cnt']

        row_sum = [sum(obs[i]) for i in range(r)]
        col_sum = [sum(obs[i][j] for i in range(r)) for j in range(c)]
        total = sum(row_sum)
        expected = [[(row_sum[i]*col_sum[j])/total for j in range(c)] for i in range(r)]

        chi2_val = 0.0
        for i in range(r):
            for j in range(c):
                e, o = expected[i][j], obs[i][j]
                if e > 0:
                    chi2_val += (o - e) ** 2 / e
        chi2 = round(chi2_val, 4)
        dof = (r - 1) * (c - 1)

        critical_005 = {1:3.8415,2:5.9915,3:7.8147,4:9.4877,5:11.0705,
                        6:12.5916,7:14.0671,8:15.5073,9:16.9190,10:18.3070}
        crit = critical_005.get(dof)
        if crit is None:
            decision = f"자유도 {dof}의 0.05 임계값이 표에 없습니다. (χ²={chi2})"
        else:
            decision = (f"χ²={chi2} > {crit:.4f} → 차이 있음"
                        if chi2 > crit else
                        f"χ²={chi2} ≤ {crit:.4f} → 차이 없음")

        crosstab = {
            'headers': ['성별 \\ 브랜드'] + brands,
            'rows': [[g] + [obs[idx_g[g]][idx_b[b]] for b in brands] for g in genders],
            'row_totals': row_sum,
            'col_totals': col_sum,
            'total': total,
        }

        return render(request, 'coffee/survey_result.html', {
            'surveys': surveys, 'decision': decision, 'chi2': chi2,
            'dof': dof, 'alpha': 0.05, 'crosstab': crosstab,
        })

    return render(request, 'coffee/survey_result.html', {
        'surveys': surveys, 'decision': result_text,
        'chi2': chi2, 'dof': dof, 'alpha': 0.05, 'crosstab': None,
    })

def survey_process(request):
    return insertDataFunc(request)

def survey_show(request):
    return survey_result(request)

def insertDataFunc(request):
    if request.method != 'POST':
        return redirect('survey_form')  # ✅ 반환 보장

    gender = request.POST.get('gender')
    age = request.POST.get('age')
    co_survey = request.POST.get('co_survey')

    if not gender or not co_survey:
        return render(request, 'coffee/survey_form.html', {
            'error': '성별과 선호 브랜드는 필수입니다.',
        })

    Survey.objects.create(
        gender=gender,
        age=int(age) if age else None,
        co_survey=co_survey,
    )
    return redirect('survey_result')    # ✅ 반환 보장
