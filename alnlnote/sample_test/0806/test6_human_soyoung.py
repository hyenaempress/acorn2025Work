# ───────────────────────────────────────────────────────────────
# test6_human.py  (C:\repository\conda\pandas_test\)
#   · human.csv   : C:\repository\human.csv
#   · tips.csv    : C:\repository\tips.csv (없으면 seaborn tips 사용)
# ───────────────────────────────────────────────────────────────
import pandas as pd
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────
base_dir   = Path(__file__).resolve().parents[2]   # C:\repository
human_path = base_dir / "human.csv"
tips_path  = base_dir / "tips.csv"

# ── (A) human.csv 읽기 ──────────────────────────────────────
try:
    # 1차 시도: 헤더 있는 경우
    human = pd.read_csv(human_path)
    human.columns = human.columns.str.strip()  # 열 이름 공백 제거
except pd.errors.ParserError:
    # 헤더가 없을 때: 직접 열 이름 부여
    human = pd.read_csv(human_path, header=None,
                        names=["Group", "Career", "Score"])
    human.columns = human.columns.str.strip()

# 열 이름이 예상과 다른 경우(예: group, GROUP 등) → 소문자로 통일
human.columns = human.columns.str.lower()

# ── (1) Group 열 결측 삭제 (소문자로 맞췄으니 'group')
human = human.dropna(subset=["group"])

# ── (2) Career·Score만 추출
human_cs = human[["career", "score"]].copy()

# ── (3) 평균
print("\n[human.csv] ───────────────────────────────────────────")
print("Career & Score 앞 5행:\n", human_cs.head())
print("\nCareer 평균 :", human_cs["career"].mean())
print("Score  평균 :", human_cs["score"].mean())

# ── (B) tips.csv 읽기 ───────────────────────────────────────
if tips_path.exists():
    tips = pd.read_csv(tips_path)
    print("\n[tips.csv] 로컬 파일 사용")
else:
    try:
        import seaborn as sns
        tips = sns.load_dataset("tips")
        print("\n[seaborn] 내장 tips 데이터 사용")
    except ModuleNotFoundError:
        print("\n⚠️  seaborn 미설치 + tips.csv 없음 → tips 작업 건너뜀")
        exit()

# ── tips 분석
print("\n[tips] info()")
print(tips.info())
print("\n앞 3행:\n", tips.head(3))
print("\n요약 통계:\n", tips.describe())
print("\n흡연자/비흡연자 수:\n", tips["smoker"].value_counts())
print("\n요일(unique):", tips["day"].unique())



