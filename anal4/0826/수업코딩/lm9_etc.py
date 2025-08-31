import joblib
import pandas as pd

# joblib 모듈 사용
# joblib.dump(lmodel, './01_big_data_machine_learning/data/mymodel.model')

# 읽기
ourmodel = joblib.load('./01_big_data_machine_learning/data/mymodel.model')

new_df = pd.DataFrame({
    'Income': [44, 44, 44],
    'Advertising': [6, 3, 11], 
    'Price': [105, 88, 77],
    'Age': [33, 55, 22]
})

new_pred = ourmodel.predict(new_df)
print("Sales 예측 결과: \n", new_pred)