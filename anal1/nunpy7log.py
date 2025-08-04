
#로그변환 : 편차가 큰 데이터를 로그변환하면 분포를 개선하고, 큰 범위의 차이를 줄이며
#모델이 보다 안정적으로 학습할 수 있도록 도와줍니다.
#로그변환은 데이터의 분포를 정규화하고, 이상치의 영향을 줄이는 데 유용합니다.
#numpy를 사용하여 로그변환을 수행할 수 있습니다.
#스케일링 이라고 말한다. 스케일링 축소 
import numpy as np

np.set_printoptions(suppress=True, precision=6)  # 출력 옵션 설정


def test(): 
    values = np.array([3.45, 34.5, 0.01, 10, 100, 1000])
    print(np.log2(3.45), np.log10(3.45), np.log(3.45))  
    print('원본자료 :'  , values   )
    log_values = np.log10(values)  #상용로그
    print('로그변환 자료 :', log_values)
    ln_values = np.log10(values)
    # 자연로그
    print('자연로그 변환 자료 :', ln_values)
    print('로그변환 자료 shape:', log_values.shape)  # (6,)
    #로그값의 최소 최대를 0~ 1사이 범위로 정규화 
    #데이터를 일정 범위에 들어오게 하는 방법
    #표준화 쵸준 편차로 요소값 마이너스 평균 (표준 편차로 나눠주는것) 값을 평균을 기준으로 분포시킴 
    #정규화 데이터의 범위를 0에서 1사이로 변환해 데이터 불포를 조정하는 방법 
    min_log = log_values.min()
    max_log = log_values.max()
    normalized = (log_values - min_log) / (max_log - min_log)
    print('정규화된 로그변환 자료:', normalized)
    
    def log_inverse():
        """로그 변환의 역변환"""
        offset = 1.0  # 로그 변환을 위한 오프셋 (0 이하 값 방지)
        log_values = np.log(10 + offset)
        original_values = np.exp(log_values) - offset  # 역변환
        print('역변환된 자료:', original_values)

    
class LogTrans:
    def __init__(self, offset=1.0):
        self.offset = offset
        # 로그 변환을 위한 오프셋 (0 이하 값 방지)
    #로그변환 수행 메소드 
    def transform(self, x: np.ndarray):
        return np.log(x + self.offset)
    
    # 역변환 메소드
    def inverse_trans(self, x: np.ndarray):
        return np.exp(x_log) - self.offset
        
    def gogo():
        print('~' * 20)
        data = np.array([0.001, 1, 10, 100, 1000,10000])
        #로그 변환용 클랙스 개체 생성 
        log_trans = LogTrans(offset=0.001)
        
        #데이터를 로그변환 하고 역뱐환을 한다 
        data_log_scaled = log_trans.transform(data)
        recovered_data =log_trans.inverse_trans(data_log_scaled)
        print('원본 데이터:', data)
        print('로그 변환된 데이터:', data_log_scaled)
        print('역변환된 데이터:', recovered_data)
           
if __name__ == "__main__":
    test()
  