# 외국인 대상 국내 주요 관광지 방문 관련 상관관계 분석
import matplotlib.pyplot as plt           # 그래프 그리기 위한 라이브러리
import matplotlib                        # matplotlib 라이브러리
import json                              # 파이썬 내장 json 라이브러리
plt.rc("font", family="Malgun Gothic")   # 한글 폰트 설정
import numpy as np                       # 수치 계산용 라이브러리
import pandas as pd                      # 데이터프레임 처리 라이브러리

# scatter Graph 작성 함수
def setScatterGraph(tour_table, all_table, tourPoint):
    # print(tourPoint)
    # 특정 관광지명에 해당하는 자료만 추출
    tour = tour_table[tour_table.resNm == tourPoint]
    # print(tour)
    # 관광지 자료와 외국인 관광 자료 병합
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    # print(merge_table)

    fig = plt.figure()                   # 그래프 Figure 생성
    fig.suptitle(tourPoint + '상관관계분석') # 그래프 제목 설정

    plt.subplot(1, 3, 1)                 # 첫 번째 subplot 생성
    plt.xlabel('중국인 입국수')           # x축 라벨
    plt.ylabel('중국인 입장객 수')        # y축 라벨
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum']) # 중국인 상관계수 계산 람다
    r1 = lamb1(merge_table)              # 상관계수 값 저장
    print('r1 :', r1)                    # 상관계수 출력
    plt.title('r={:5f}'.format(r1))      # subplot 제목에 상관계수 표시
    plt.scatter(merge_table['china'], merge_table['ForNum'], alpha=0.7, s=6, c='red') # 산점도 그리기

    plt.subplot(1, 3, 2)                 # 두 번째 subplot 생성
    plt.xlabel('일본인 입국수')           # x축 라벨
    plt.ylabel('일본인 입장객 수')        # y축 라벨
    lamb1 = lambda p:merge_table['japan'].corr(merge_table['ForNum']) # 일본인 상관계수 계산 람다
    r2 = lamb1(merge_table)              # 상관계수 값 저장
    print('r2 :', r2)                    # 상관계수 출력
    plt.title('r={:5f}'.format(r2))      # subplot 제목에 상관계수 표시
    plt.scatter(merge_table['japan'], merge_table['ForNum'], alpha=0.7, s=6, c='blue') # 산점도 그리기

    plt.subplot(1, 3, 3)                 # 세 번째 subplot 생성
    plt.xlabel('미국인 입국수')           # x축 라벨
    plt.ylabel('미국인 입장객 수')        # y축 라벨
    lamb1 = lambda p:merge_table['usa'].corr(merge_table['ForNum']) # 미국인 상관계수 계산 람다
    r3 = lamb1(merge_table)              # 상관계수 값 저장
    print('r3 :', r3)                    # 상관계수 출력
    plt.title('r={:5f}'.format(r3))      # subplot 제목에 상관계수 표시
    plt.scatter(merge_table['usa'], merge_table['ForNum'], alpha=0.7, s=6, c='green') # 산점도 그리기

    plt.tight_layout()                   # subplot 간격 자동 조정
    plt.show()                           # 그래프 표시
    
    return [tourPoint, r1, r2, r3]       # 관광지명과 각국 상관계수 반환

def chulbal():
    # 서울시 관광지 정보를 읽어서 DataFrame으로 저장
    fname = "./01_big_data_machine_learning/data/서울특별시_관광지입장정보_2011_2016.json"
    jsonTP = json.loads(open(fname, 'r', encoding='utf-8').read()) # json 파일 읽기
    tour_table = pd.DataFrame(jsonTP, columns = ('yyyymm', 'resNm', 'ForNum')) # 데이터프레임 생성
    tour_table = tour_table.set_index(['yyyymm'])                  # yyyymm을 인덱스로 설정
    # print(tour_table)

    resNm = tour_table.resNm.unique()                              # 관광지명 목록 추출
    print('resNm :', resNm[:5])                                    # 관광지명 일부 출력

    # 중국인 관광 정보를 읽어 DataFrame으로 저장
    cdf = "./01_big_data_machine_learning/data/중국인방문객.json"
    jdata = json.loads(open(cdf, 'r', encoding='utf-8').read())    # json 파일 읽기
    china_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt')) # 데이터프레임 생성
    china_table = china_table.rename(columns={'visit_cnt': 'china'})   # 컬럼명 변경
    china_table = china_table.set_index('yyyymm')                      # yyyymm을 인덱스로 설정
    # print("\n", china_table[:2])

    # 일본인 관광 정보를 읽어 DataFrame으로 저장
    cdf = "./01_big_data_machine_learning/data/일본인방문객.json"
    jdata = json.loads(open(cdf, 'r', encoding='utf-8').read())    # json 파일 읽기
    japan_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt')) # 데이터프레임 생성
    japan_table = japan_table.rename(columns={'visit_cnt': 'japan'})   # 컬럼명 변경
    japan_table = japan_table.set_index('yyyymm')                      # yyyymm을 인덱스로 설정
    # print("\n", japan_table[:2])

    # 미국인 관광 정보를 읽어 DataFrame으로 저장
    cdf = "./01_big_data_machine_learning/data/미국인방문객.json"
    jdata = json.loads(open(cdf, 'r', encoding='utf-8').read())    # json 파일 읽기
    usa_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))   # 데이터프레임 생성
    usa_table = usa_table.rename(columns={'visit_cnt': 'usa'})         # 컬럼명 변경
    usa_table = usa_table.set_index('yyyymm')                          # yyyymm을 인덱스로 설정
    # print("\n", usa_table[:2])

    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True) # 중국, 일본 데이터 병합
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)     # 미국 데이터 병합

    r_list = []                                                     # 결과 저장 리스트
    for tourPoint in resNm[:5]:                                     # 관광지명 5개 반복
        # print(tourPoint)
        # 각 관광지별 상관계수와 그래프 그리기
        r_list.append(setScatterGraph(tour_table, all_table, tourPoint))

    # r_list로 DataFrame 작성
    r_df = pd.DataFrame(r_list, columns=('고궁명', '중국', '일본', '미국')) # 결과 데이터프레임 생성
    print(r_df)                                                     # 결과 출력
    r_df = r_df.set_index('고궁명')                                  # 고궁명을 인덱스로 설정
    print(r_df)                                                     # 결과 출력

    r_df.plot(kind='bar', rot=50)                                   # 막대그래프 그리기
    plt.show()                                                      # 그래프 표시
    plt.close()                                                     # 그래프 닫기

if __name__ == "__main__":
    chulbal()                                                       # 메인 함수 실행