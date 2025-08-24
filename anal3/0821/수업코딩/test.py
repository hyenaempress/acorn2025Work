import pandas as pd
import numpy as np

data = pd.read_csv("C:/Users/rlarb/Desktop/data/07-2_add_label(randomized pick)/02_evaluation/label.csv")

print(data)
data.columns = [
        '절댓값 평균값_1', '제곱 평균값_1', '표준편차_1', '중앙값_1',
        '절댓값 평균값_2', '제곱 평균값_2', '표준편차_2', '중앙값_2',
        '절댓값 평균값_3', '제곱 평균값_3', '표준편차_3', '중앙값_3',
        '절댓값 평균값_4', '제곱 평균값_4', '표준편차_4', '중앙값_4',
        '절댓값 평균값_5', '제곱 평균값_5', '표준편차_5', '중앙값_5',
        '절댓값 평균값_6', '제곱 평균값_6', '표준편차_6', '중앙값_6',
        'labels'
]

print(data)

label_0_data = data[data['labels'] == 0]
label_1_data = data[data['labels'] == 1]
label_2_data = data[data['labels'] == 2]
print(label_0_data)
