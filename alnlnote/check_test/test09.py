from pandas import DataFrame, Series
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
df = DataFrame(data)
results = Series([x.split()[0] for x in df.juso])
print(results)
