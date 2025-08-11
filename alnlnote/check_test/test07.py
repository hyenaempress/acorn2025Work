from pandas import DataFrame
frame = DataFrame({'bun':[1,2,3,4], 'irum':['aa','bb','cc','dd']}, index=['a','b', 'c','d'])
print(frame)
frame2 = frame.drop('d')
print(frame2)

