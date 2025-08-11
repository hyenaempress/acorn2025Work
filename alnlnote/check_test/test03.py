# [문항3] sqlite3 DB를 사용하여 DataFrame의 자료를 db에 저장하려 한다. 아래의 빈칸에 알맞은 코드를 적으시오.
# 조건 : index는 저장에서 제외한다. (배점:5)
# data = {
#     'product':['아메리카노','카페라떼','카페모카'],
#     'maker':['스벅','이디아','엔젤리너스'],
#     'price':[5000,5500,6000]
# }

# df = pd.DataFrame(data)
# df.to_sql('test', conn, if_exists='append', index=False)