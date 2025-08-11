import numpy as np
import pandas as pd

df = pd.DataFrame(np.arange(12).reshape((4, 3)), 
                  index=['1월', '2월', '3월', '4월'], 
                  columns=['강남', '강북', '서초'])

print(df)