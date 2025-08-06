import pandas as pd
import numpy as np

# 데이터 생성
print("=== 학생 데이터 생성 중... ===")
n_rows = 10000

data = {
    'id': range(1, n_rows + 1),
    'name': [f'Student_{i}' for i in range(1, n_rows + 1)],
    'score1': np.random.randint(50, 101, size=n_rows),
    'score2': np.random.randint(50, 101, size=n_rows)
}

df = pd.DataFrame(data)

# 데이터 미리보기
print("=== 데이터 미리보기 ===")
print("첫 3행:")
print(df.head(3))
print("\n마지막 3행:")
print(df.tail(3))

# 데이터 정보
print(f"\n=== 데이터 정보 ===")
print(f"총 행 수: {len(df):,}")
print(f"열 수: {len(df.columns)}")
print(f"데이터 타입:")
print(df.dtypes)

# 기본 통계
print(f"\n=== 기본 통계 ===")
print(df.describe())

# CSV 파일로 저장
csv_path = 'students.csv'  # 수정: student.csv -> students.csv
df.to_csv(csv_path, index=False)  # 수정: to_scv -> to_csv

print(f"\n=== 파일 저장 완료 ===")
print(f"파일명: {csv_path}")
print(f"파일 크기: 약 {len(df) * len(df.columns) * 10 / 1024:.1f} KB")

# 저장된 파일 확인
print(f"\n=== 저장된 파일 확인 ===")
try:
    saved_df = pd.read_csv(csv_path)
    print(f"저장된 파일 행 수: {len(saved_df):,}")
    print("저장된 파일의 첫 5행:")
    print(saved_df.head())
    print("\n✅ 파일이 성공적으로 저장되었습니다!")
except Exception as e:
    print(f"❌ 파일 저장 확인 중 오류: {e}")

# 샘플 데이터 몇 개 보기
print(f"\n=== 랜덤 샘플 5개 ===")
print(df.sample(5))