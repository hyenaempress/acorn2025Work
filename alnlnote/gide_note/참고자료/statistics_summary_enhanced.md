# 통계 분석 요약 (0814) - 보완 및 강화 버전

본 문서는 ANOVA, 가설검정 구조, 카이제곱 검정(CV vs p-value 포함) 핵심 정리.

---

## 2. 가설 검정 기본 구조

### 2.1 용어
- 귀무가설(H₀): 차이·효과·관계 없음
- 대립가설(H₁): 차이·효과·관계 있음

### 2.2 판단
- 유의수준 α (기본 0.05)
- p-value < α → H₀ 기각 / p-value ≥ α → H₀ 기각 못함

### 2.3 예시
- H₀: μ_개 = μ_고양이 / H₁: μ_개 ≠ μ_고양이 (t-test 또는 ANOVA)

### **2.4 오류의 종류 (추가)**
| 오류 | 실제 상황 | 판단 | 결과 |
|------|-----------|------|------|
| Type I (α) | H₀ 참 | H₀ 기각 | 거짓 양성 |
| Type II (β) | H₀ 거짓 | H₀ 채택 | 거짓 음성 |
| 검정력 | 1-β | H₁ 참일 때 올바른 기각 | 민감도 |

---

## 3. ANOVA(분산분석)

### 3.1 목적
여러 집단 평균 차이 검정 (F = 집단간분산 / 집단내분산)

### 3.2 종류
- 3.2.1 일원 ANOVA: 한 요인
- 3.2.2 이원 ANOVA: 두 요인 + 상호작용
- 3.2.3 반복측정 ANOVA: 동일 개체 반복 측정
- 3.2.4 이원 반복측정 ANOVA: 2요인 + 반복

### 3.3 가정
정규성, 등분산성, 독립성(반복측정은 구형성)

### 3.4 사후검정
전체 유의 → Tukey 등으로 어떤 집단 차이인지 탐색

### **3.5 ANOVA 코드 예시 (추가)**
```python
from scipy import stats
import pandas as pd

# 일원 ANOVA
group1 = [23, 25, 27, 29]
group2 = [30, 32, 34, 36] 
group3 = [20, 22, 24, 26]
f_stat, p_value = stats.f_oneway(group1, group2, group3)

# 이원 ANOVA (statsmodels 사용)
import statsmodels.api as sm
from statsmodels.formula.api import ols

# df는 'value', 'factor1', 'factor2' 컬럼을 가진 DataFrame
model = ols('value ~ factor1 + factor2 + factor1:factor2', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
```

---

## 4. 카이제곱(χ²) 검정 개념

### 4.1 목적
범주형 관측도수 vs 기대도수 차이 평가

### 4.2 종류
- 4.2.1 적합도(Goodness of Fit)
- 4.2.2 독립성(Independence)
- 4.2.3 동질성(Homogeneity)

### 4.3 절차
1) 교차표 생성
2) 기대도수 E = (행합×열합)/전체합
3) χ² = Σ (O−E)² / E
4) 자유도 df = (행−1)(열−1)
5) p-value 또는 임계값으로 판단

### 4.4 해석
- χ² 크거나 p-value ≤ α → H₀ 기각(관계/차이 있음)
- 그렇지 않음 → H₀ 기각 못함(관계 증거 부족)

### 4.5 예시 가설
- H₀: 벼락치기 공부와 합격 여부 무관
- H₁: 관련 있음

### **4.6 카이제곱 검정 조건 (추가)**
- **필수 조건**: 모든 기대빈도 ≥ 5
- **권장 조건**: 기대빈도 < 5인 셀이 전체의 20% 미만
- **대안**: Fisher의 정확검정 (작은 표본)

---

## **5. 효과크기 (새로 추가)**

### 5.1 중요성
통계적 유의성 ≠ 실무적 중요성

### 5.2 ANOVA 효과크기
- **η² (에타 제곱)**: SS_between / SS_total
- **Cohen's d**: (μ₁ - μ₂) / σ_pooled
- **해석**: 작음(0.01), 중간(0.06), 큼(0.14)

### 5.3 카이제곱 효과크기
- **Cramér's V**: √(χ²/(n×min(r-1,c-1)))
- **해석**: 작음(0.1), 중간(0.3), 큼(0.5)

```python
# 효과크기 계산 예시
import numpy as np

# Cramér's V
def cramers_v(chi2, n, r, c):
    return np.sqrt(chi2 / (n * min(r-1, c-1)))

# 사용 예
v = cramers_v(chi2_stat, total_n, num_rows, num_cols)
```

---

## 6. 코드 예시 (독립성 검정)

```python
import pandas as pd, scipy.stats as stats
data = pd.read_csv("./01_big_data_machine_learning/data/pass_cross.csv")
ctab = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
chi2, p, dof, expected = stats.chi2_contingency(ctab)
print(chi2, p, dof)
```

예시 결과: χ²=3.0, df=1, p=0.5578 → p>0.05 → H₀ 기각 못함.

---

## 7. p-value 개념

### 7.1 정의
H₀ 참 가정 하에 관측된 통계량 이상 나올 확률

### 7.2 특징
작을수록(≤α) H₀ 기각 근거 강함

### 7.3 계산
라이브러리 자동 (scipy.stats.*)

---

## 8. 임계값(CV) vs p-value 비교

### 8.1 동등성
두 방식은 동일 결론 도출 (형식만 다름)

### 8.2 정의
- CV: df, α로 분포표에서 찾은 경계
- p-value: P(Χ²(df) ≥ 관측 χ²)

### 8.3 결정 규칙
| 방식 | 기각 조건 | 기각 못함 |
|------|-----------|-----------| 
| CV   | χ² ≥ CV   | χ² < CV   |
| p    | p ≤ α     | p > α     |

### 8.4 p-value 선호 이유
표 불필요, α 변경 유연

### 8.5 계산 개요
1) O 정리 2) E 계산 3) χ² 합산 4) df 산출 5) 오른쪽 꼬리 확률

### 8.6 직관
(O−E) 괴리↑ → χ²↑ → p↓ → H₀ 기각 근거↑

### 8.7 예시
χ²=14.2, df=5 → p≈0.014 <0.05 → H₀ 기각

### 8.8 한 줄 요약
CV 비교와 p-value 비교는 논리 동일, p-value가 실무 표준.

---

## **9. 다중비교 문제 (새로 추가)**

### 9.1 문제점
여러 검정 시 Type I 오류율 증가

### 9.2 보정 방법
- **Bonferroni**: α_adj = α/m (보수적)
- **Holm**: 순차적 Bonferroni
- **FDR**: False Discovery Rate 제어

```python
from statsmodels.stats.multitest import multipletests

# 예시: 여러 p-values 보정
p_values = [0.01, 0.04, 0.03, 0.50]
rejected, p_adj, _, _ = multipletests(p_values, method='bonferroni')
```

---

## 10. 전처리·실무 팁

### 10.1 데이터 정리
- 결측 제거: `df = df.dropna(subset=['col1','col2'])`
- 범주 매핑: `df['직급코드'] = df['직급'].replace({'이사':1,'부장':2,...})`
- 구간화: `pd.cut / pd.qcut`

### 10.2 검정 조건 확인
- 기대도수 체크: `(expected < 5).sum()==0` 권장
- 정규성 검정: `stats.shapiro()`, `stats.normaltest()`
- 등분산성 검정: `stats.levene()`, `stats.bartlett()`

### 10.3 주의사항
- 독립성 ≠ 인과관계
- 표본크기와 검정력 고려
- 실무적 의미 해석 병행

---

## **11. 비모수 대안 (새로 추가)**

### 11.1 ANOVA 대안
- **Kruskal-Wallis**: 일원 ANOVA 비모수 버전
- **Friedman**: 반복측정 ANOVA 비모수 버전

### 11.2 카이제곱 대안
- **Fisher의 정확검정**: 작은 표본
- **G-test**: 우도비 검정

```python
# 비모수 검정 예시
from scipy.stats import kruskal, friedmanchisquare
from scipy.stats.contingency import fisher_exact

# Kruskal-Wallis
h_stat, p = kruskal(group1, group2, group3)

# Fisher 정확검정 (2x2 표)
odds_ratio, p = fisher_exact([[8, 2], [1, 5]])
```

---

## 12. 추가 코드 스니펫

### 12.1 적합도(주사위)
```python
import scipy.stats as stats
obs = [4,6,17,16,8,9]; exp=[10]*6
print(stats.chisquare(obs, exp))
```

### 12.2 선호도(음료)
```python
import pandas as pd, scipy.stats as stats
url="https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinkdata.csv"
df = pd.read_csv(url)
print(stats.chisquare(df['관측도수']))
```

### 12.3 범주형 교차표 시각화(예시)
```python
import seaborn as sns, matplotlib.pyplot as plt
sns.heatmap(ctab.iloc[:-1,:-1], annot=True, fmt='d', cmap='Blues')
plt.show()
```

### **12.4 종합 분석 파이프라인 (추가)**
```python
def comprehensive_analysis(data, factor_col, outcome_col):
    """종합적인 통계 분석 파이프라인"""
    
    # 1. 기술통계
    print("=== 기술통계 ===")
    print(data.groupby(factor_col)[outcome_col].describe())
    
    # 2. 정규성 검정
    from scipy.stats import shapiro
    for group in data[factor_col].unique():
        group_data = data[data[factor_col]==group][outcome_col]
        stat, p = shapiro(group_data)
        print(f"{group} 정규성: p={p:.4f}")
    
    # 3. 등분산성 검정
    from scipy.stats import levene
    groups = [data[data[factor_col]==g][outcome_col] for g in data[factor_col].unique()]
    stat, p = levene(*groups)
    print(f"등분산성: p={p:.4f}")
    
    # 4. ANOVA 또는 비모수 검정
    if p > 0.05:  # 등분산성 만족
        from scipy.stats import f_oneway
        stat, p = f_oneway(*groups)
        print(f"ANOVA: F={stat:.4f}, p={p:.4f}")
    else:
        from scipy.stats import kruskal
        stat, p = kruskal(*groups)
        print(f"Kruskal-Wallis: H={stat:.4f}, p={p:.4f}")
    
    # 5. 효과크기
    # (구현 생략)
    
    return stat, p
```

---

## 13. 해석 요약 표

| 검정 | 질문 | 통계량 | 기각 기준(p) | 비모수 대안 |
|------|------|--------|--------------|-------------|
| ANOVA | 평균 차이? | F | p ≤ α | Kruskal-Wallis |
| χ² 적합도 | 분포 일치? | χ² | p ≤ α | G-test |
| χ² 독립성/동질성 | 관계/분포 동일? | χ² | p ≤ α | Fisher 정확검정 |

---

## **14. 실무 체크리스트 (새로 추가)**

### 14.1 분석 전 확인사항
- [ ] 연구 질문 명확화
- [ ] 적절한 검정 방법 선택
- [ ] 표본크기 충분성 확인
- [ ] 가정 조건 검토

### 14.2 분석 중 확인사항  
- [ ] 결측치 처리
- [ ] 이상치 탐지
- [ ] 가정 검정 실시
- [ ] 적절한 유의수준 설정

### 14.3 분석 후 확인사항
- [ ] 효과크기 계산
- [ ] 다중비교 보정 (필요시)
- [ ] 실무적 의미 해석
- [ ] 결과의 일반화 가능성 검토

---

## 15. 주의 사항

### 15.1 해석 주의점
- p ≥ α → H₀가 '참' 입증 아님 (근거 부족)
- 기대도수 너무 작으면 (특히 <5) χ² 근사 약화 → Fisher 등 고려
- 통계적 유의성 ≠ 실질 효과 (효과크기 별도)

### **15.2 윤리적 고려사항 (추가)**
- p-hacking 방지
- 선택적 보고 금지
- 전체 분석 과정 투명성 유지

---

## 16. 핵심 용어

- 자유도(df): 독립 정보 수
- 기대도수(expected): H₀ 하 이론적 빈도
- 사후검정(post-hoc): ANOVA 유의 후 세부 비교
- **검정력(Power)**: 실제 효과가 있을 때 이를 탐지할 확률 (추가)
- **효과크기(Effect Size)**: 통계적 유의성과 별개의 실질적 중요성 (추가)

---

## 17. 참고 자료

- SciPy: chisquare, chi2_contingency
- Statsmodels: ANOVA, 사후검정, 다중비교
- Wikipedia: ANOVA, Chi-squared test
- **추가 권장**: Cohen's Statistical Power Analysis, Field's Discovering Statistics